import os
import uuid
import io
from typing import List, Dict, Any

from dotenv import load_dotenv
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import faiss
import gradio as gr

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------------------------
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
NEO4J_URI = os.getenv("NEO4J_AURA_URI")
NEO4J_USERNAME = os.getenv("NEO4J_AURA_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_AURA_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_AURA_DATABASE", "neo4j")  # default is 'neo4j'

# ---------------------------------------------------------------------------
# Module-level state — services initialised lazily on first PDF upload
# ---------------------------------------------------------------------------
_state: Dict[str, Any] = {
    "faiss_index": None,
    "index_to_chunk_id": None,
    "all_chunk_data": None,
    "ready": False,
    # services (filled by _init_services)
    "graph": None,
    "llm": None,
    "embedding_model": None,
    "services_ready": False,
}


def _init_services():
    """Connect to Neo4j, load LLM and embedding model (called once on first use)."""
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
    _state["embedding_model"] = SentenceTransformer("clip-ViT-B-32")
    _state["services_ready"] = True


# ===========================================================================
# Pipeline functions
# ===========================================================================

def process_pdf(pdf_path: str) -> List[Dict]:
    """Extract text and images from every page of a PDF."""
    doc = fitz.open(pdf_path)
    processed_data = []

    for page_num, page in enumerate(doc):
        text = page.get_text()
        images = []
        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)
            images.append(Image.open(io.BytesIO(base_image["image"])))

        processed_data.append({
            "page_num": page_num + 1,
            "text": text,
            "images": images,
        })

    doc.close()
    return processed_data


def create_graph_from_chunks(data: List[Dict]) -> List[Dict]:
    """Chunk text, build a knowledge graph in Neo4j, return chunks with images."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=50)
    all_chunks_with_images: List[Dict] = []

    for page_data in data:
        if not page_data["text"].strip():
            continue

        chunks = text_splitter.split_text(page_data["text"])
        for chunk in chunks:
            chunk_id = str(uuid.uuid4())
            doc = Document(
                page_content=chunk,
                metadata={"source_page": page_data["page_num"], "chunk_id": chunk_id},
            )
            try:
                transformer = LLMGraphTransformer(llm=_state["llm"])
                graph_documents = transformer.convert_to_graph_documents([doc])

                for node in graph_documents[0].nodes:
                    node.properties["source_chunk_id"] = chunk_id
                for rel in graph_documents[0].relationships:
                    rel.properties["source_chunk_id"] = chunk_id

                _state["graph"].add_graph_documents(graph_documents)
                all_chunks_with_images.append({
                    "chunk_id": chunk_id,
                    "text": chunk,
                    "images": page_data["images"],
                })
            except Exception as e:
                print(f"Error processing a chunk: {e}")

    return all_chunks_with_images


def find_and_merge_duplicates():
    """Merge duplicate nodes in Neo4j based on their 'id' property."""
    merge_query = """
    MATCH (n)
    WITH n.id AS nodeId, collect(n) AS nodes
    WHERE size(nodes) > 1
    CALL apoc.refactor.mergeNodes(nodes, {mergeRels: true}) YIELD node
    RETURN count(*) AS merged_count
    """
    try:
        result = _state["graph"].query(merge_query)
        return result[0]["merged_count"] if result else 0
    except Exception as e:
        print(f"Error during node merging (APOC may not be installed): {e}")
        return 0


def create_faiss_index(chunks: List[Dict]):
    """Build a FAISS index from text + image embeddings."""
    embeddings = []
    index_to_chunk_id = {}
    idx = 0

    for chunk_data in chunks:
        text_emb = _state["embedding_model"].encode(chunk_data["text"])
        embeddings.append(text_emb)
        index_to_chunk_id[idx] = chunk_data["chunk_id"]
        idx += 1

        for image in chunk_data["images"]:
            img_emb = _state["embedding_model"].encode(image)
            embeddings.append(img_emb)
            index_to_chunk_id[idx] = chunk_data["chunk_id"]
            idx += 1

    matrix = np.array(embeddings).astype("float32")
    faiss_index = faiss.IndexFlatL2(matrix.shape[1])
    faiss_index.add(matrix)
    return faiss_index, index_to_chunk_id


def query_pipeline(query: str, k: int = 3):
    """Search FAISS + Neo4j and return (text_results, images)."""
    query_emb = _state["embedding_model"].encode([query]).astype("float32")
    distances, indices = _state["faiss_index"].search(query_emb, k)

    retrieved_ids = list(set(
        _state["index_to_chunk_id"][i] for i in indices[0]
    ))

    # Collect unique images from matching chunks
    unique_images: List[Image.Image] = []
    seen = set()
    for chunk in _state["all_chunk_data"]:
        if chunk["chunk_id"] in retrieved_ids:
            for img in chunk["images"]:
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                h = hash(buf.getvalue())
                if h not in seen:
                    unique_images.append(img)
                    seen.add(h)

    # Retrieve subgraph from Neo4j
    cypher = """
    MATCH (n) WHERE n.source_chunk_id IN $chunk_ids
    OPTIONAL MATCH (n)-[r]-(m)
    RETURN n, r, m
    """
    graph_results = _state["graph"].query(cypher, params={"chunk_ids": retrieved_ids})

    return graph_results, unique_images


# ===========================================================================
# Gradio callback functions
# ===========================================================================

def handle_pdf_upload(pdf_file) -> str:
    """Process an uploaded PDF through the full pipeline."""
    if pdf_file is None:
        return "Please upload a PDF file first."

    try:
        yield "Initialising services (Neo4j, LLM, embeddings)..."
        _init_services()

        yield "Step 1/4  Extracting text and images from PDF..."
        pdf_data = process_pdf(pdf_file.name)
        page_count = len(pdf_data)
        image_count = sum(len(p["images"]) for p in pdf_data)

        yield f"Step 2/4  Building knowledge graph ({page_count} pages, {image_count} images)..."
        all_chunks = create_graph_from_chunks(pdf_data)

        yield "Step 3/4  Merging duplicate nodes..."
        merged = find_and_merge_duplicates()

        yield "Step 4/4  Creating FAISS embeddings index..."
        faiss_idx, idx_map = create_faiss_index(all_chunks)

        # Store in module state
        _state["faiss_index"] = faiss_idx
        _state["index_to_chunk_id"] = idx_map
        _state["all_chunk_data"] = all_chunks
        _state["ready"] = True

        yield (
            f"Done!\n"
            f"  Pages processed: {page_count}\n"
            f"  Images extracted: {image_count}\n"
            f"  Text chunks created: {len(all_chunks)}\n"
            f"  FAISS vectors: {faiss_idx.ntotal}\n"
            f"  Duplicate nodes merged: {merged}"
        )
    except Exception as e:
        yield f"Error: {e}"


def handle_query(user_query: str):
    """Run a search query and return formatted text + images."""
    if not _state["ready"]:
        return "Please upload and process a PDF first (use the 'Upload PDF' tab).", []

    if not user_query.strip():
        return "Please enter a query.", []

    try:
        graph_results, images = query_pipeline(user_query)

        # Format text output
        lines = [f"Query: {user_query}", f"Graph paths retrieved: {len(graph_results)}", ""]
        for i, record in enumerate(graph_results, 1):
            lines.append(f"--- Result {i} ---")
            for key, val in record.items():
                lines.append(f"  {key}: {val}")
            lines.append("")

        if not graph_results:
            lines.append("No matching nodes found in the knowledge graph.")

        text_output = "\n".join(lines)
        return text_output, images

    except Exception as e:
        return f"Error during query: {e}", []


# ===========================================================================
# Gradio UI
# ===========================================================================

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Multimodal GraphRAG") as app:
        gr.Markdown("# Multimodal GraphRAG\nUpload a PDF to build a knowledge graph, then query it.")

        with gr.Tab("Upload PDF"):
            pdf_input = gr.File(label="Choose a PDF", file_types=[".pdf"])
            process_btn = gr.Button("Process PDF", variant="primary")
            status_box = gr.Textbox(label="Status", lines=8, interactive=False)

            process_btn.click(fn=handle_pdf_upload, inputs=pdf_input, outputs=status_box)

        with gr.Tab("Query"):
            query_input = gr.Textbox(label="Enter your question", placeholder="e.g. What are the key findings?")
            search_btn = gr.Button("Search", variant="primary")
            results_box = gr.Textbox(label="Results", lines=12, interactive=False)
            image_gallery = gr.Gallery(label="Retrieved Images", columns=3, height="auto")

            search_btn.click(fn=handle_query, inputs=query_input, outputs=[results_box, image_gallery])

    return app


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="127.0.0.1", server_port=7860)

