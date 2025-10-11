GOOGLE_API_KEY="AIzaSyDym4fxzRIB5kSiMANltRSLpED_MKUSGiQ"

# This cell contains the setup code (PDF processing, graph creation, FAISS index creation)
# It only needs to be run once.

import uuid
from dotenv import load_dotenv
from typing import List, Dict, Any

# File processing
import fitz  # PyMuPDF
from PIL import Image
import io

# LangChain components
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph

# Embeddings and Vector Store
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load environment variables from .env file
load_dotenv()


NEO4J_URI="neo4j+s://61986679.databases.neo4j.io"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="bpPcx73PRuDgRVptDJ9jZYAYsG0NhbAFr3CRjVlWAJE"

# Instantiate the Neo4j graph connection
graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD
)

# Instantiate the LLM for graph transformation
# Using a specific model version for reproducibility
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

# Instantiate the CLIP model for embeddings
# This will download the model on the first run
embedding_model = SentenceTransformer('clip-ViT-B-32')

# --- 1. PDF Data Extraction ---
def process_pdf(pdf_path: str) -> List[Dict]:
    """
    Extracts text and images from a PDF file and associates them.
    Each element in the output list represents a page.
    """
    doc = fitz.open(pdf_path)
    processed_data = []
    print(f"Processing PDF: {pdf_path} with {len(doc)} pages.")

    for page_num, page in enumerate(doc):
        # Extract text
        text = page.get_text()

        # Extract images
        images = []
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            images.append(image)

        processed_data.append({
            "page_num": page_num + 1,
            "text": text,
            "images": images
        })
        print(f"  - Extracted text and {len(images)} images from page {page_num + 1}.")

    doc.close()
    return processed_data

# --- 2. Chunking and Graph Creation ---
def create_graph_from_chunks(data: List[Dict]):
    """
    Chunks text, creates graph documents, and adds them to Neo4j.
    Returns the raw text chunks with unique IDs and associated images.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=50)

    all_chunks_with_images = []

    for page_data in data:
        page_text = page_data["text"]
        if not page_text.strip():
            continue

        # Chunk the text from the page
        chunks = text_splitter.split_text(page_text)

        for chunk in chunks:
            chunk_id = str(uuid.uuid4())
            # Create a Document object for LangChain
            doc = Document(
                page_content=chunk,
                metadata={"source_page": page_data["page_num"], "chunk_id": chunk_id}
            )

            # Extract graph data from the document
            try:
                llm_transformer=LLMGraphTransformer(llm=llm)
                graph_documents = llm_transformer.convert_to_graph_documents([doc])
                print(f"  - Generated {len(graph_documents[0].nodes)} nodes and {len(graph_documents[0].relationships)} relationships for a chunk.")

                # Add a 'source_chunk_id' property to all nodes and relationships
                for node in graph_documents[0].nodes:
                    node.properties["source_chunk_id"] = chunk_id
                for rel in graph_documents[0].relationships:
                    rel.properties["source_chunk_id"] = chunk_id


                # Add to Neo4j
                graph.add_graph_documents(graph_documents)
                all_chunks_with_images.append({
                    "chunk_id": chunk_id,
                    "text": chunk,
                    "images": page_data["images"] # Associate images with the chunk
                    })


            except Exception as e:
                print(f"Error processing a chunk: {e}")

    return all_chunks_with_images

# --- 3. Deduplication and Entity Disambiguation Setup ---
def find_and_merge_duplicates(graph: Neo4jGraph):
    """
    Finds and merges nodes that are likely duplicates based on their 'id' property.
    This is a basic approach; more sophisticated methods might be needed for complex cases.
    """
    print("\nAttempting to find and merge duplicate nodes...")
    # This query finds nodes with the same 'id' and merges them, keeping the oldest node.
    # It also merges relationships connected to the duplicate nodes.
    merge_query = """
    MATCH (n)
    WITH n.id AS nodeId, collect(n) AS nodes
    WHERE size(nodes) > 1
    CALL apoc.refactor.mergeNodes(nodes, {mergeRels: true}) YIELD node
    RETURN count(*) AS merged_count
    """
    try:
        result = graph.query(merge_query)
        merged_count = result[0]['merged_count'] if result else 0
        print(f"Merged {merged_count} sets of duplicate nodes based on 'id'.")
    except Exception as e:
        print(f"Error during node merging: {e}")
        print("Please ensure you have the APOC plugin installed and enabled in your Neo4j database.")


# def identify_potential_disambiguation_candidates(graph: Neo4jGraph) -> Dict[str, Dict[str, Any]]:
#     """
#     Identifies potential entities that might require disambiguation.
#     This basic version finds nodes with similar sounding IDs or that appear in related chunks.
#     Returns a dictionary where keys are potential entity names/IDs and values are details.
#     """
#     print("\nIdentifying potential entity disambiguation candidates...")
#     # This query finds nodes that share a source_chunk_id, indicating they appeared in the same chunk.
#     # It can be a starting point for identifying entities that might be related or duplicates.
#     candidate_query = """
#     MATCH (n)
#     WHERE n.source_chunk_id IS NOT NULL
#     WITH n.source_chunk_id AS chunkId, collect({id: n.id, label: labels(n)[0], properties: properties(n)}) AS nodes_in_chunk
#     WHERE size(nodes_in_chunk) > 1
#     RETURN chunkId, nodes_in_chunk
#     """

#     candidates_by_chunk = graph.query(candidate_query)
#     potential_candidates = {}

#     for record in candidates_by_chunk:
#         chunk_id = record['chunkId']
#         nodes_in_chunk = record['nodes_in_chunk']
#         for node_info in nodes_in_chunk:
#             node_id = node_info['id']
#             if node_id not in potential_candidates:
#                 potential_candidates[node_id] = {
#                     "label": node_info.get('label', 'Unknown'),
#                     "source_chunk_ids": [chunk_id],
#                     "properties": node_info.get('properties', {})
#                 }
#             else:
#                 if chunk_id not in potential_candidates[node_id]["source_chunk_ids"]:
#                     potential_candidates[node_id]["source_chunk_ids"].append(chunk_id)

#     print(f"Identified {len(potential_candidates)} potential entity disambiguation candidates.")
#     # In a real scenario, you'd do more sophisticated analysis here (e.g., string similarity)
#     return potential_candidates


# --- 4. Embeddings and FAISS Indexing ---
def create_faiss_index(chunks: List[Dict]):
    """
    Creates multimodal embeddings for text and images and stores them in FAISS.
    Returns the FAISS index and a mapping from FAISS index to chunk_id.
    """
    embeddings = []
    index_to_chunk_id = {}
    current_index = 0

    print("\nGenerating embeddings and building FAISS index...")
    for chunk_data in chunks:
        # 1. Embed the text chunk
        text_embedding = embedding_model.encode(chunk_data["text"])
        embeddings.append(text_embedding)
        index_to_chunk_id[current_index] = chunk_data["chunk_id"]
        current_index += 1

        # 2. Embed associated images (if any)
        for image in chunk_data["images"]:
            image_embedding = embedding_model.encode(image)
            embeddings.append(image_embedding)
            index_to_chunk_id[current_index] = chunk_data["chunk_id"]
            current_index += 1

    # Convert list of embeddings to a numpy array
    embedding_matrix = np.array(embeddings).astype('float32')

    # Create a FAISS index
    dimension = embedding_matrix.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embedding_matrix)

    print(f"FAISS index created with {faiss_index.ntotal} vectors.")
    return faiss_index, index_to_chunk_id

# --- 5. Querying Pipeline ---
def query_pipeline(query: str, faiss_index, index_to_chunk_id: Dict, all_chunks_data: List[Dict], k: int = 3):
    """
    Searches FAISS for relevant chunks, retrieves corresponding subgraphs from Neo4j,
    and also retrieves the associated images.
    Returns retrieved graph data and a list of unique images associated with the retrieved chunks.
    """
    print(f"\n--- Running Query: '{query}' ---")

    # 1. FAISS Search
    query_embedding = embedding_model.encode([query]).astype('float32')
    distances, indices = faiss_index.search(query_embedding, k)

    # Get the unique chunk IDs from the top k results
    retrieved_chunk_ids = list(set([index_to_chunk_id[i] for i in indices[0]]))
    print(f"FAISS search found relevant chunk IDs: {retrieved_chunk_ids}")

    # 2. Retrieve associated images based on retrieved_chunk_ids
    retrieved_images = []
    for chunk_data in all_chunks_data:
        if chunk_data["chunk_id"] in retrieved_chunk_ids:
            retrieved_images.extend(chunk_data["images"])

    # Get unique images to avoid duplicates if multiple chunks from the same page are retrieved
    unique_images = []
    seen_images = set()
    for img in retrieved_images:
      img_bytes = io.BytesIO()
      img.save(img_bytes, format="PNG")
      img_hash = hash(img_bytes.getvalue())
      if img_hash not in seen_images:
        unique_images.append(img)
        seen_images.add(img_hash)


    # 3. Neo4j Search
    cypher_query = """
    MATCH (n) WHERE n.source_chunk_id IN $chunk_ids
    OPTIONAL MATCH (n)-[r]-(m)
    RETURN n, r, m
    """
    results = graph.query(cypher_query, params={"chunk_ids": retrieved_chunk_ids})

    if not results:
        print("No matching nodes found in Neo4j for the retrieved chunks.")

    print(f"Retrieved {len(results)} paths from Neo4j.")
    print(f"Retrieved {len(unique_images)} unique images associated with the retrieved chunks.")

    return results, unique_images


PDF_FILE_PATH = "sample.pdf"

# Step 1: Process the PDF to get text and images per page
pdf_data = process_pdf(PDF_FILE_PATH)

# Step 2: Create graph from chunks and ingest into Neo4j
all_chunk_data = create_graph_from_chunks(pdf_data)

# Step 3: Deduplication and Entity Disambiguation Setup
# Attempt to merge duplicate nodes based on their 'id'
find_and_merge_duplicates(graph)

# # Identify potential candidates for more complex entity disambiguation (e.g., using LLM)
# potential_disambiguation_entities = identify_potential_disambiguation_candidates(graph)
# print("\nPotential disambiguation candidates identified. Further processing with LLM may be needed.")


# Step 4: Create FAISS index with multimodal embeddings
if all_chunk_data:
    # Re-create FAISS index after potential node merges
    faiss_index, index_to_chunk_id_map = create_faiss_index(all_chunk_data)
    print("\nSetup complete. You can now use the query cell below.")
else:
    print("No chunks were processed, skipping embedding and querying.")



user_query = input("Enter your query: ")

# Run the query pipeline
if 'faiss_index' in locals() and 'index_to_chunk_id_map' in locals() and 'all_chunk_data' in locals():
    retrieved_graph_data, retrieved_images = query_pipeline(
        query=user_query,
        faiss_index=faiss_index,
        index_to_chunk_id=index_to_chunk_id_map,
        all_chunks_data=all_chunk_data
    )

    # Print the retrieved graph data
    print("\n--- Final Retrieved Graph Data ---")
    for record in retrieved_graph_data:
        print(record)

    # Display the retrieved images
    print("\n--- Retrieved Images ---")
    if retrieved_images:
        for i, img in enumerate(retrieved_images):
            print(f"Image {i+1}:")
            
    else:
        print("No images retrieved for the relevant chunks.")
else:
    print("Please run the setup cell first.")