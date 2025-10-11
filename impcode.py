# import re
# import requests
# import os
# import torch
# import clip
# from PIL import Image
# import pandas as pd
# import faiss
# import requests
# from langchain_ollama.llms import OllamaLLM
# from langchain_core.prompts import ChatPromptTemplate
# from pathlib import Path
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)
# BASE_DIR = Path(__file__).resolve().parent
# import numpy as np
# csv_path = os.path.join(BASE_DIR, 'static', 'product4.csv')
# df = pd.read_csv(csv_path)

# def get_clip_embedding(row):
#     text = f"{row['product_name']}.{row['category']}.{row['description']}"
    
#     # Clean image path
#     image_path =os.path.join(BASE_DIR, 'static', re.sub(r'_\d+(?=\.)', '', row['image_path']))
    
#     # Open image and preprocess
#     image = Image.open(image_path).convert("RGB")  # Ensure 3 channels
#     image = preprocess(image).unsqueeze(0).to(device)  # Preprocess for CLIP

#     # Tokenize text
#     text_tokens = clip.tokenize([text]).to(device)
    
#     # Get embeddings
#     with torch.no_grad():
#         image_features = model.encode_image(image)
#         text_features = model.encode_text(text_tokens)
#         combined = (image_features + text_features) / 2
#         return combined.cpu().numpy()

# embeddings = [get_clip_embedding(row) for idx, row in df.iterrows()]

# dimension = embeddings[0].shape[1]
# index = faiss.IndexFlatL2(dimension)
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)
# # Wrap with metadata
# product_ids = df['product_id'].tolist()
# index_with_ids = faiss.IndexIDMap(index)
# index_with_ids.add_with_ids(np.vstack(embeddings), np.array(product_ids))

# from neo4j import GraphDatabase

# # Neo4j connection (driver)
# NEO4J_URI = "bolt://localhost:7687"
# NEO4J_USER = "neo4j"
# NEO4J_PASSWORD = "secretgraph"



# def get_clip_embedding(row):
#     text = f"{row['product_name']}.{row['category']}.{row['description']}"
    
  
#     image_path =os.path.join(BASE_DIR, 'static', re.sub(r'_\d+(?=\.)', '', row['image_path']))
#     image = Image.open(image_path).convert("RGB")
#     image = preprocess(Image.open(image_path)).unsqueeze(0)
#     text_tokens = clip.tokenize([text]).to(device)

#     with torch.no_grad():
#         image_features = model.encode_image(image)
#         text_features = model.encode_text(text_tokens)
#         combined = (image_features + text_features) / 2
#         return combined.cpu().numpy()

# embeddings = [get_clip_embedding(row) for idx, row in df.iterrows()]

# driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
# def fetch_prod(product_ids):
#     results = []
#     for pid in product_ids:
#         record = driver.execute_query(
#             """
#             MATCH (p:Product {productId: $pid})
#             OPTIONAL MATCH (p)-[:BELONGS_TO]->(c:Category)
#             OPTIONAL MATCH (p)-[:HAS_PRICE]->(pr:Price)
#             OPTIONAL MATCH (p)-[:HAS_IMAGE]->(img:Image)
#             RETURN p.name AS name, p.productId AS id,
#                    c.name AS category, pr.value AS price,
#                    img.path AS imagePath
#             """,
#             pid=int(pid)
#         )
#         if record and record.records:
#             results.append(record.records[0].data())
#     return results
# def embed_query(text_query):
#     text_tokens = clip.tokenize([text_query]).to(device)
#     with torch.no_grad():
#         text_features = model.encode_text(text_tokens)
#     return text_features.cpu().numpy()

# def search_faiss(query_embedding, k=5):
#     D, I = index_with_ids.search(query_embedding, k)
#     return I[0].tolist()  # List of productIds

# def fetch_prod(product_ids):
#     results = []
#     for pid in product_ids:
#         record = driver.execute_query(
#             """
#             MATCH (p:Product {productId: $pid})
#             OPTIONAL MATCH (p)-[:BELONGS_TO]->(c:Category)
#             OPTIONAL MATCH (p)-[:HAS_PRICE]->(pr:Price)
#             OPTIONAL MATCH (p)-[:HAS_IMAGE]->(img:Image)
#             RETURN p.name AS name, p.productId AS id,
#                    c.name AS category, pr.value AS price,
#                    img.path AS imagePath
#             """,
#             pid=int(pid)
#         )
#         if record and record.records:
#             results.append(record.records[0].data())
#     return results
# def getNodes(product_info):
#     results2 = []
#     for pid in product_info:
#         record = driver.execute_query(
#             """
#             MATCH (startNode:Product {name: $nm})-[*1..3]-(endNode)   
#             RETURN DISTINCT endNode
#             """,
#             nm=pid['name']
#         )
#         if record and record.records:
#             results2.append(record.records[0].data())
#     return results2
# def semantic_product_search(user_query):
#     query_embedding = embed_query(user_query)
#     product_ids = search_faiss(query_embedding)
#     product_info = fetch_prod(product_ids)
#     return product_info
# def semantic_productdfs(user_query):
#     query_embedding = embed_query(user_query)
#     product_ids = search_faiss(query_embedding)
#     product_info = fetch_prod(product_ids)
#     print(product_info)
#     for r in product_info:
#       print(r)
#     res=getNodes(product_info)
#     return res
        



# # model1 = OllamaLLM(model="llama3.2:1b")
# # def search_query:
# # template = """
# # You are a product recommendation assistant.
# # You will receive:
# # - a list of product nodes with their details (name, category, price, imagePath)
# # - a list of related nodes or connections
# # - a natural-language question from the user

# # Use only this information to answer.
# # Format your final response exactly like this:

# # Query: <the original question>
# # Here are a few options:
# # 1. <Product Name> ₹<Price> Category: <Category>
# #    Image: <Image Path>
# # 2. <Product Name> ₹<Price> Category: <Category>
# #    Image: <Image Path>

# # If no relevant products exist, say "No matching options found based on the available nodes."
# # Do not include commentary or explanations outside this format.

# # Nodes: {reviews}
# # Connections: {bfs}
# # Question: {question}
# # """


# # prompt = ChatPromptTemplate.from_template(template)
# # chain = prompt | model1

# # while True:
# #     print("\n\n-------------------------------")
# #     question = input("Ask your question (q to quit): ")
# #     print("\n\n")
# #     if question == "q":
# #         break
# #     bfs=semantic_productdfs(question)
# #     reviews = semantic_product_search(question)
# #     print (reviews)
# #     result = chain.invoke({"reviews": reviews, "question": question,"bfs":bfs})
# #     print(result)
# model1 = OllamaLLM(model="llama3.2:1b")

# def search_query():
#     template = """
#     You are a product recommendation assistant.
#     You will receive:
#     - a list of product nodes with their details (name, category, price, imagePath)
#     - a list of related nodes or connections
#     - a natural-language question from the user

#     Use only this information to answer.
#     Format your final response exactly like this:

#     Query: <the original question>
#     Here are a few options:
#     1. <Product Name> ₹<Price> Category: <Category>
#        Image: <Image Path>
#     2. <Product Name> ₹<Price> Category: <Category>
#        Image: <Image Path>

#     If no relevant products exist, say "No matching options found based on the available nodes."
#     Do not include commentary or explanations outside this format.

#     Nodes: {reviews}
#     Connections: {bfs}
#     Question: {question}
#     """

#     prompt = ChatPromptTemplate.from_template(template)
#     chain = prompt | model1

#     while True:
#         print("\n\n-------------------------------")
#         question = input("Ask your question (q to quit): ")
#         print("\n\n")
#         if question.lower() == "q":
#             break

#         bfs = semantic_productdfs(question)
#         reviews = semantic_product_search(question)

#         print(reviews)
#         result = chain.invoke({
#             "reviews": reviews,
#             "question": question,
#             "bfs": bfs
#         })
#         print(result)    
import os
import re
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
model, preprocess = clip.load("ViT-B/32", device=device)

# -------------------------------
# Paths
# -------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = os.path.join(BASE_DIR, "static", "product4.csv")

# -------------------------------
# Load CSV
# -------------------------------
df = pd.read_csv(CSV_PATH)

# -------------------------------
# Neo4j driver
# -------------------------------
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "secretgraph"
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# -------------------------------
# Functions
# -------------------------------
def get_clip_embedding(row):
    """Generate combined CLIP embedding for image + text."""
    text = f"{row['product_name']}.{row['category']}.{row['description']}"

    # Clean image path
    image_path = os.path.join(BASE_DIR, 'static', re.sub(r'_\d+(?=\.)', '', row['image_path']))

    if not os.path.exists(image_path):
        print(f"⚠️ Missing image: {image_path}")
        return None

    # Preprocess image
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0).to(device)

    # Tokenize text
    text_tokens = clip.tokenize([text]).to(device)

    # Generate embeddings
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)
        combined = (image_features + text_features) / 2
        return combined.cpu().numpy()


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
    raise ValueError("No valid embeddings found. Check your images.")

# -------------------------------
# Setup FAISS index
# -------------------------------
dimension = embeddings[0].shape[1]
index = faiss.IndexFlatL2(dimension)
index_with_ids = faiss.IndexIDMap(index)
index_with_ids.add_with_ids(np.vstack(embeddings), np.array(valid_product_ids))



import torch
import clip
from PIL import Image
import pandas as pd
import faiss

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)





import re

def get_clip_embedding(row):
    # Remove _number suffix from product_name
    sanitized_name = re.sub(r'_\d+$', '', row['product_name'])
    text = f"{sanitized_name}.{row['category']}.{row['description']}"
    text_tokens = clip.tokenize([text]).to(device)
    
    image_path = row['image_path']
    image_path = re.sub(r'(_\d+)(\.\w+)$', r'\2', row['image_path'])
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)

        if os.path.exists(image_path):
            try:
                image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
                image_features = model.encode_image(image)
                combined = (image_features + text_features) / 2
                return combined.cpu().numpy()
            except Exception as e:
                print(f"Skipping image for {row['product_name']} due to error: {e}")
                return text_features.cpu().numpy()
        else:
            print(f"Image not found for {row['product_name']}, using text only.")
            return text_features.cpu().numpy()

embeddings = [get_clip_embedding(row) for idx, row in df.iterrows()]

import numpy as np

dimension = embeddings[0].shape[1]
index = faiss.IndexFlatL2(dimension)
product_ids = df['product_id'].tolist()
index_with_ids = faiss.IndexIDMap(index)
index_with_ids.add_with_ids(np.vstack(embeddings), np.array(product_ids))

from neo4j import GraphDatabase

# Neo4j connection (driver)
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "secretgraph"


driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
# Clear existing data
driver.execute_query("MATCH (n) DETACH DELETE n")
# Insert products into Neo4j
for idx, row in df.iterrows():
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



# -------------------------------
# Neo4j / FAISS Helper Functions
# -------------------------------
def embed_query(text_query):
    tokens = clip.tokenize([text_query]).to(device)
    with torch.no_grad():
        return model.encode_text(tokens).cpu().numpy()

def search_faiss(query_embedding, k=5):
    D, I = index_with_ids.search(query_embedding, k)
    return I[0].tolist()

def fetch_prod(product_ids):
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

def getNodes(product_info):
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
            results.append(record.records[0].data())
    return results

def semantic_product_search(user_query):
    query_embedding = embed_query(user_query)
    product_ids = search_faiss(query_embedding)
    product_info = fetch_prod(product_ids)
    return product_info

def semantic_product_dfs(user_query):
    query_embedding = embed_query(user_query)
    product_ids = search_faiss(query_embedding)
    product_info = fetch_prod(product_ids)
    res = getNodes(product_info)
    return res

# -------------------------------
# Ollama / LangChain Setup
# -------------------------------
model1 = OllamaLLM(model="llama3.2:1b")

def run_query_chain(question, reviews, bfs):
    """Run the LangChain template for a single query."""
    template = """
    You are a product recommendation assistant.
    You will receive:
    - a list of product nodes with their details (name, category, price, imagePath)
    - a list of related nodes or connections
    - a natural-language question from the user

    Use only this information to answer.
    Give a paragraph answer based on the input
    Query: <the original question>
      Here are a few options:
    1. {{<Product Name>}} ₹{{<Price>}} Category: {{<Category>}}
       Image: {{<Image Path>}}
    2. {{<Product Name>}} ₹{{<Price>}} Category: {{<Category>}}
       Image: {{<Image Path>}}
    Do not include commentary or explanations outside give a paragraph answer based on this  

    Nodes: {reviews}
    Connections: {bfs}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model1
    result = chain.invoke({"reviews": reviews, "question": question, "bfs": bfs})
    return result


print(f"Total products processed: {len(valid_product_ids)}")
print(f"Embedding dimension: {embeddings[0].shape[1] if embeddings else 'No embeddings'}")
