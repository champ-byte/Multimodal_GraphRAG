import os
import re
import sys
from pathlib import Path

import torch
import clip
from PIL import Image
import pandas as pd
import numpy as np
import faiss
from neo4j import GraphDatabase
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# -------------------------------
# Set device and load CLIP model
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# -------------------------------
# Paths
# -------------------------------
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = os.path.join(BASE_DIR, "static", "product4.csv")

# -------------------------------
# Load CSV
# -------------------------------
df = pd.read_csv(CSV_PATH)

# -------------------------------
# Neo4j driver
# -------------------------------
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "secretgraph")

try:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print(" Connected to Neo4j successfully.")
except Exception as e:
    print(f" Failed to connect to Neo4j: {e}")
    sys.exit(1)


# -------------------------------
# Functions
# -------------------------------
def get_clip_embedding(row):
    """Generate combined CLIP embedding for image + text.
    Falls back to text-only embedding if the image is missing or corrupted.
    """
    sanitized_name = re.sub(r'_\d+$', '', row['product_name'])
    text = f"{sanitized_name}.{row['category']}.{row['description']}"

    # Clean image path
    image_path = os.path.join(BASE_DIR, 'static', re.sub(r'_\d+(?=\.)', '', row['image_path']))

    # Tokenize text
    text_tokens = clip.tokenize([text], truncate=True).to(device)

    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)

        if os.path.exists(image_path):
            try:
                image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
                image_features = clip_model.encode_image(image)
                combined = (image_features + text_features) / 2
                return combined.cpu().numpy()
            except Exception as e:
                print(f"⚠️ Skipping image for {row['product_name']} due to error: {e}")
                return text_features.cpu().numpy()
        else:
            print(f"⚠️ Image not found for {row['product_name']}, using text only.")
            return text_features.cpu().numpy()


# -------------------------------
# Generate embeddings for all products
# -------------------------------
embeddings = []
valid_product_ids = []

for idx, row in df.iterrows():
    e = get_clip_embedding(row)
    if e is not None:
        embeddings.append(e)
        valid_product_ids.append(row['product_id'])

if not embeddings:
    raise ValueError("No valid embeddings found. Check your images and CSV data.")

# -------------------------------
# Setup FAISS index
# -------------------------------
dimension = embeddings[0].shape[1]
index = faiss.IndexFlatL2(dimension)
index_with_ids = faiss.IndexIDMap(index)
index_with_ids.add_with_ids(np.vstack(embeddings), np.array(valid_product_ids))


# -------------------------------
# Neo4j data reset (with confirmation)
# -------------------------------
def reset_neo4j_data():
    """Clears all existing data from Neo4j and re-inserts products from the CSV."""
    confirm = input("⚠️  This will DELETE ALL existing data in Neo4j. Continue? (yes/no): ").strip().lower()
    if confirm != "yes":
        print("Skipped database reset.")
        return False

    driver.execute_query("MATCH (n) DETACH DELETE n")
    print("🗑️  Cleared existing Neo4j data.")

    for _, row in df.iterrows():
        driver.execute_query(
            """
            MERGE (p:Product {productId: $pid, name: $name})
            MERGE (c:Category {name: $category})
            MERGE (pr:Price {value: $price})
            MERGE (img:Image {path: $imagePath})

            MERGE (p)-[:BELONGS_TO]->(c)
            MERGE (p)-[:HAS_PRICE]->(pr)
            MERGE (p)-[:HAS_IMAGE]->(img)
            """,
            pid=int(row['product_id']),
            name=row['product_name'],
            category=row['category'],
            price=float(row['price']),
            imagePath=row['image_path']
        )
    print(f"✅ Inserted {len(df)} products into Neo4j.")
    return True


# -------------------------------
# Neo4j / FAISS Helper Functions
# -------------------------------
def embed_query(text_query):
    """Embed a text query using CLIP."""
    tokens = clip.tokenize([text_query], truncate=True).to(device)
    with torch.no_grad():
        return clip_model.encode_text(tokens).cpu().numpy()


def search_faiss(query_embedding, k=5):
    """Search FAISS index for nearest neighbors."""
    D, I = index_with_ids.search(query_embedding, k)
    return I[0].tolist()


def fetch_prod(product_ids):
    """Fetch product details from Neo4j by product IDs."""
    results = []
    for pid in product_ids:
        record = driver.execute_query(
            """
            MATCH (p:Product {productId: $pid})
            OPTIONAL MATCH (p)-[:BELONGS_TO]->(c:Category)
            OPTIONAL MATCH (p)-[:HAS_PRICE]->(pr:Price)
            OPTIONAL MATCH (p)-[:HAS_IMAGE]->(img:Image)
            RETURN p.name AS name, p.productId AS id,
                   c.name AS category, pr.value AS price,
                   img.path AS imagePath
            """,
            pid=int(pid)
        )
        if record and record.records:
            results.append(record.records[0].data())
    return results


def get_nodes(product_info):
    """Traverse the graph up to 3 hops from each product and return all connected nodes."""
    results = []
    for pid in product_info:
        record = driver.execute_query(
            """
            MATCH (startNode:Product {name: $nm})-[*1..3]-(endNode)
            RETURN DISTINCT endNode
            """,
            nm=pid['name']
        )
        if record and record.records:
            for r in record.records:
                results.append(r.data())
    return results


def semantic_product_search(user_query):
    """Find products matching a text query via CLIP + FAISS."""
    query_embedding = embed_query(user_query)
    product_ids = search_faiss(query_embedding)
    return fetch_prod(product_ids)


def semantic_product_dfs(user_query):
    """Find products and their graph neighbors matching a text query."""
    query_embedding = embed_query(user_query)
    product_ids = search_faiss(query_embedding)
    product_info = fetch_prod(product_ids)
    return get_nodes(product_info)


# -------------------------------
# Ollama / LangChain Setup
# -------------------------------
llm_model = OllamaLLM(model="llama3.2:1b")

TEMPLATE = """
You are a product recommendation assistant.
You will receive:
- a list of product nodes with their details (name, category, price, imagePath)
- a list of related nodes or connections
- a natural-language question from the user

Use only this information to answer.
Give a paragraph answer based on the input.

Query: the original question
Here are a few options:
1. <Product Name> ₹<Price> Category: <Category>
   Image: <Image Path>
2. <Product Name> ₹<Price> Category: <Category>
   Image: <Image Path>

Do not include commentary or explanations outside the paragraph answer.

Nodes: {reviews}
Connections: {bfs}
Question: {question}
"""


def run_query_chain(question, reviews, bfs):
    """Run the LangChain template for a single query."""
    prompt = ChatPromptTemplate.from_template(TEMPLATE)
    chain = prompt | llm_model
    result = chain.invoke({"reviews": reviews, "question": question, "bfs": bfs})
    return result


# -------------------------------
# Main entry point
# -------------------------------
def main():
    print(f"Total products processed: {len(valid_product_ids)}")
    print(f"Embedding dimension: {embeddings[0].shape[1] if embeddings else 'No embeddings'}")

    # Prompt user to reset Neo4j data
    reset_neo4j_data()

    # Interactive query loop
    while True:
        print("\n-------------------------------")
        question = input("Ask your question (q to quit): ")
        if question.strip().lower() == "q":
            break

        reviews = semantic_product_search(question)
        bfs = semantic_product_dfs(question)
        result = run_query_chain(question, reviews, bfs)
        print("\n" + result)

    driver.close()
    print("👋 Session ended.")


if __name__ == "__main__":
    main()
