import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Any

import torch
import clip
from PIL import Image
import pandas as pd
import numpy as np
import faiss
import gradio as gr

from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------------
# Load environment variables (same .env as kavyacode.py)
# ---------------------------------------------------------------------------
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
NEO4J_URI = os.getenv("NEO4J_AURA_URI")
NEO4J_USERNAME = os.getenv("NEO4J_AURA_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_AURA_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_AURA_DATABASE", "neo4j")

# ---------------------------------------------------------------------------
# CLIP model
# ---------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# ---------------------------------------------------------------------------
# Module-level state — populated lazily during pipeline execution
# ---------------------------------------------------------------------------
_state: Dict[str, Any] = {
    "graph": None,
    "llm": None,
    "services_ready": False,
    # Data populated after CSV processing
    "df": None,
    "faiss_index": None,
    "valid_row_ids": None,
    "embeddings": None,
    "node_columns": [],
    "relationships": [],
    "ready": False,
}

BASE_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Service initialisation
# ---------------------------------------------------------------------------
def _init_services():
    """Connect to Neo4j Aura and load Gemini LLM (called once)."""
    if _state["services_ready"]:
        return
    try:
        _state["graph"] = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            database=NEO4J_DATABASE,
        )
    except Exception as e:
        raise RuntimeError(
            f"Could not connect to Neo4j.\n"
            f"Check NEO4J_AURA_URI / credentials in your .env file.\n\nDetail: {e}"
        )
    _state["llm"] = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    _state["services_ready"] = True


# ===========================================================================
# CLIP helpers
# ===========================================================================

def _clip_text_embedding(text: str) -> np.ndarray:
    """Embed a text string with CLIP and return a (1, D) numpy array."""
    tokens = clip.tokenize([text], truncate=True).to(device)
    with torch.no_grad():
        return clip_model.encode_text(tokens).cpu().numpy()


def _clip_row_embedding(row: pd.Series, node_columns: List[str]) -> np.ndarray:
    """Build a combined CLIP embedding for one CSV row.

    Concatenates the text of all selected node columns and, if an 'image_path'
    column exists and the image is on disk, averages text + image embeddings.
    """
    text = ". ".join(str(row[c]) for c in node_columns if pd.notna(row[c]))
    text_emb = _clip_text_embedding(text)

    # Try to incorporate an image if available
    if "image_path" in row.index and pd.notna(row.get("image_path")):
        img_path = os.path.join(BASE_DIR, "static", str(row["image_path"]))
        img_path = re.sub(r'_\d+(?=\.)', '', img_path)  # strip numeric suffix
        if os.path.exists(img_path):
            try:
                image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
                with torch.no_grad():
                    img_emb = clip_model.encode_image(image).cpu().numpy()
                return (text_emb + img_emb) / 2
            except Exception:
                pass  # fall through to text-only

    return text_emb


# ===========================================================================
# Graph-building helpers
# ===========================================================================

def _sanitize_label(name: str) -> str:
    """Turn a column name into a valid Neo4j label (PascalCase, no spaces)."""
    return "".join(word.capitalize() for word in re.split(r"[\s_]+", name.strip()))


def _insert_graph_data(
    df: pd.DataFrame,
    node_columns: List[str],
    relationships: List[Dict],
):
    """Insert nodes and relationships into Neo4j based on user-selected schema."""
    graph = _state["graph"]

    # 1. Create nodes for every unique value in each node column
    for col in node_columns:
        label = _sanitize_label(col)
        unique_vals = df[col].dropna().unique()
        for val in unique_vals:
            cypher = f"MERGE (n:{label} {{value: $val}})"
            graph.query(cypher, params={"val": str(val)})

    # 2. Create relationships
    for rel in relationships:
        src_col = rel["source"]
        tgt_col = rel["target"]
        rel_label = re.sub(r"[^A-Z0-9_]", "_", rel["label"].upper().strip())
        src_label = _sanitize_label(src_col)
        tgt_label = _sanitize_label(tgt_col)

        for _, row in df.iterrows():
            src_val = row.get(src_col)
            tgt_val = row.get(tgt_col)
            if pd.isna(src_val) or pd.isna(tgt_val):
                continue
            cypher = (
                f"MATCH (a:{src_label} {{value: $src}}), (b:{tgt_label} {{value: $tgt}}) "
                f"MERGE (a)-[:{rel_label}]->(b)"
            )
            graph.query(cypher, params={"src": str(src_val), "tgt": str(tgt_val)})


# ===========================================================================
# Query helpers
# ===========================================================================

def _search_faiss(query_text: str, k: int = 5) -> List[int]:
    """Return row indices of the k nearest neighbours."""
    q_emb = _clip_text_embedding(query_text).astype("float32")
    D, I = _state["faiss_index"].search(q_emb, k)
    return [int(i) for i in I[0] if i >= 0]


def _fetch_graph_context(row_ids: List[int]) -> str:
    """Fetch subgraph context from Neo4j for the matched rows."""
    df = _state["df"]
    node_columns = _state["node_columns"]
    graph = _state["graph"]
    lines = []

    for rid in row_ids:
        if rid >= len(df):
            continue
        row = df.iloc[rid]
        # For each node column value, find its 1-3 hop neighbourhood
        for col in node_columns:
            val = row.get(col)
            if pd.isna(val):
                continue
            label = _sanitize_label(col)
            cypher = (
                f"MATCH (start:{label} {{value: $val}})-[*1..3]-(end) "
                f"RETURN DISTINCT start.value AS start, labels(end)[0] AS endLabel, end.value AS endValue "
                f"LIMIT 20"
            )
            try:
                results = graph.query(cypher, params={"val": str(val)})
                for r in results:
                    lines.append(f"{r['start']} -> ({r['endLabel']}) {r['endValue']}")
            except Exception:
                pass

    return "\n".join(lines) if lines else "No graph connections found."


TEMPLATE = """\
You are a knowledgeable assistant.
You will receive:
- A list of relevant rows from a CSV dataset.
- A graph-context showing how entities are connected.
- A natural-language question from the user.

Use only this information to answer. Give a clear, concise paragraph answer.

Relevant Rows:
{rows}

Graph Context:
{graph_context}

Question: {question}
"""


def _run_llm(question: str, rows_text: str, graph_context: str) -> str:
    """Send context + question to Gemini and return the answer."""
    prompt = ChatPromptTemplate.from_template(TEMPLATE)
    chain = prompt | _state["llm"]
    return chain.invoke({
        "rows": rows_text,
        "graph_context": graph_context,
        "question": question,
    }).content


# ===========================================================================
# Gradio callbacks
# ===========================================================================

def handle_csv_upload(csv_file):
    """Parse CSV and return column names for the configuration step."""
    if csv_file is None:
        return gr.update(), gr.update(), gr.update(), gr.update(), [], "Please upload a CSV file."

    try:
        df = pd.read_csv(csv_file.name)
        _state["df"] = df
        columns = list(df.columns)
        preview = df.head(5).to_string(index=False)

        return (
            gr.update(choices=columns, value=[], visible=True),   # node checkboxes
            gr.update(choices=columns, visible=True),             # source dropdown
            gr.update(choices=columns, visible=True),             # target dropdown
            gr.update(visible=True),                              # label + add btn row
            [],                                                   # reset rel_state
            f"Loaded CSV with {len(df)} rows and {len(columns)} columns:\n"
            f"{', '.join(columns)}\n\nPreview:\n{preview}",
        )
    except Exception as e:
        return gr.update(), gr.update(), gr.update(), gr.update(), [], f"Error reading CSV: {e}"


def _add_relationship(src, tgt, lbl, current_rels):
    """Add a relationship definition to the state list."""
    if not src or not tgt or not lbl or not lbl.strip():
        return current_rels, _rels_to_display(current_rels), "Fill in all three fields before adding."
    entry = {"source": src, "target": tgt, "label": lbl.strip()}
    updated = current_rels + [entry]
    return updated, _rels_to_display(updated), f"Added: {src} --[{lbl.strip()}]--> {tgt}"


def _remove_last_relationship(current_rels):
    """Remove the last relationship from the state list."""
    if not current_rels:
        return current_rels, _rels_to_display(current_rels), "Nothing to remove."
    removed = current_rels[-1]
    updated = current_rels[:-1]
    return updated, _rels_to_display(updated), f"Removed: {removed['source']} --[{removed['label']}]--> {removed['target']}"


def _rels_to_display(rels):
    """Convert relationship list to a Dataframe for display."""
    if not rels:
        return pd.DataFrame(columns=["#", "Source", "Relationship", "Target"])
    rows = []
    for i, r in enumerate(rels, 1):
        rows.append({"#": i, "Source": r["source"], "Relationship": r["label"], "Target": r["target"]})
    return pd.DataFrame(rows)


def handle_build_graph(node_cols, rel_state):
    """Build FAISS index + Neo4j graph from the uploaded CSV."""
    if _state["df"] is None:
        yield "Upload a CSV first."
        return
    if not node_cols:
        yield "Select at least one node column."
        return

    df = _state["df"]
    relationships = rel_state if rel_state else []

    _state["node_columns"] = node_cols
    _state["relationships"] = relationships

    try:
        yield "Initialising services (Neo4j Aura + Gemini)..."
        _init_services()

        yield f"Step 1/3 -- Generating CLIP embeddings for {len(df)} rows..."
        embeddings = []
        valid_ids = []
        for idx, row in df.iterrows():
            emb = _clip_row_embedding(row, node_cols)
            if emb is not None:
                embeddings.append(emb)
                valid_ids.append(idx)

        if not embeddings:
            yield "No valid embeddings could be generated."
            return

        matrix = np.vstack(embeddings).astype("float32")
        dimension = matrix.shape[1]
        faiss_index = faiss.IndexFlatL2(dimension)
        faiss_index.add(matrix)

        _state["faiss_index"] = faiss_index
        _state["valid_row_ids"] = valid_ids
        _state["embeddings"] = embeddings

        yield f"Step 2/3 -- Inserting {len(node_cols)} node types and {len(relationships)} relationship types into Neo4j..."
        _insert_graph_data(df, node_cols, relationships)

        yield "Step 3/3 -- Finalising..."
        _state["ready"] = True

        node_count = sum(df[c].dropna().nunique() for c in node_cols)
        yield (
            f"Graph built successfully!\n"
            f"  Rows processed: {len(df)}\n"
            f"  FAISS vectors: {faiss_index.ntotal}\n"
            f"  Unique nodes created: ~{node_count}\n"
            f"  Relationship types: {len(relationships)}\n\n"
            f"Switch to the Query tab to search."
        )
    except Exception as e:
        yield f"Error: {e}"


def _collect_images(matched_indices: List[int]) -> List[Image.Image]:
    """Load images from static/ for matched CSV rows that have an image_path column."""
    df = _state["df"]
    images = []
    seen_paths = set()

    if "image_path" not in df.columns:
        return images

    for rid in matched_indices:
        if rid >= len(df):
            continue
        img_rel = df.iloc[rid].get("image_path")
        if pd.isna(img_rel):
            continue
        img_path = os.path.join(BASE_DIR, "static", str(img_rel))
        if img_path in seen_paths:
            continue
        seen_paths.add(img_path)
        if os.path.exists(img_path):
            try:
                images.append(Image.open(img_path).convert("RGB"))
            except Exception:
                pass

    return images


def handle_query(user_query: str):
    """Search FAISS + Neo4j and generate an LLM answer."""
    if not _state["ready"]:
        return "Please upload a CSV and build the graph first (use the 'Upload & Configure' tab).", []

    if not user_query.strip():
        return "Please enter a query.", []

    try:
        # FAISS search
        matched_indices = _search_faiss(user_query, k=5)
        df = _state["df"]
        node_cols = _state["node_columns"]

        # Build rows text
        matched_rows = df.iloc[matched_indices]
        rows_text = matched_rows[node_cols].to_string(index=False)

        # Graph context
        graph_context = _fetch_graph_context(matched_indices)

        # Collect images for matched rows
        images = _collect_images(matched_indices)

        # LLM answer
        answer = _run_llm(user_query, rows_text, graph_context)

        text_result = (
            f"**Query:** {user_query}\n\n"
            f"**Answer:**\n{answer}\n\n"
            f"---\n"
            f"**Matched rows ({len(matched_indices)}):**\n```\n{rows_text}\n```\n\n"
            f"**Graph context:**\n```\n{graph_context}\n```"
        )
        return text_result, images
    except Exception as e:
        return f"Error during query: {e}", []


# ===========================================================================
# Gradio UI
# ===========================================================================

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="CSV GraphRAG") as app:
        gr.Markdown(
            "# CSV GraphRAG\n"
            "Upload any CSV, choose which columns become **nodes** and define **relationships**, "
            "then query your auto-built knowledge graph with natural language."
        )

        # Gradio State to hold the list of relationship dicts
        rel_state = gr.State([])

        # ---- Tab 1: Upload & Configure ----
        with gr.Tab("Upload & Configure"):
            with gr.Row():
                csv_input = gr.File(label="Upload CSV", file_types=[".csv"])

            status_box = gr.Textbox(label="Status", lines=6, interactive=False)

            gr.Markdown("### Select Node Columns")
            gr.Markdown("Pick which CSV columns should become **graph nodes**.")
            node_checkboxes = gr.CheckboxGroup(
                choices=[], label="Node Columns", visible=False
            )

            gr.Markdown("### Define Relationships")
            gr.Markdown(
                "Select a **source column**, a **target column**, type a **label**, "
                "and click **Add**. Repeat for as many relationships as you need."
            )
            with gr.Row():
                rel_src_dd = gr.Dropdown(choices=[], label="Source Column", visible=False)
                rel_tgt_dd = gr.Dropdown(choices=[], label="Target Column", visible=False)
            with gr.Row(visible=False) as rel_action_row:
                rel_label_txt = gr.Textbox(label="Relationship Label", placeholder="e.g. BELONGS_TO")
                add_rel_btn = gr.Button("Add", variant="primary", scale=0)
                rm_rel_btn = gr.Button("Remove Last", variant="secondary", scale=0)

            rel_feedback = gr.Textbox(label="", interactive=False, lines=1, visible=True, show_label=False)
            rel_display = gr.Dataframe(
                headers=["#", "Source", "Relationship", "Target"],
                datatype=["number", "str", "str", "str"],
                label="Defined Relationships",
                interactive=False,
                visible=True,
            )

            build_btn = gr.Button("Build Graph", variant="primary")
            build_status = gr.Textbox(label="Build Status", lines=8, interactive=False)

            # Wire events
            csv_input.change(
                fn=handle_csv_upload,
                inputs=csv_input,
                outputs=[node_checkboxes, rel_src_dd, rel_tgt_dd, rel_action_row, rel_state, status_box],
            )

            add_rel_btn.click(
                fn=_add_relationship,
                inputs=[rel_src_dd, rel_tgt_dd, rel_label_txt, rel_state],
                outputs=[rel_state, rel_display, rel_feedback],
            )

            rm_rel_btn.click(
                fn=_remove_last_relationship,
                inputs=[rel_state],
                outputs=[rel_state, rel_display, rel_feedback],
            )

            build_btn.click(
                fn=handle_build_graph,
                inputs=[node_checkboxes, rel_state],
                outputs=build_status,
            )

        # ---- Tab 2: Query ----
        with gr.Tab("Query"):
            query_input = gr.Textbox(
                label="Ask a question",
                placeholder="e.g. Which products are in the electronics category?",
            )
            search_btn = gr.Button("Search", variant="primary")
            results_box = gr.Markdown(label="Results")
            image_gallery = gr.Gallery(label="Retrieved Images", columns=3, height="auto")

            search_btn.click(
                fn=handle_query,
                inputs=query_input,
                outputs=[results_box, image_gallery],
            )

    return app


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="127.0.0.1", server_port=7861)

