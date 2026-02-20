# Multimodal_GraphRAG

[DOCUMENTATION](https://manaswibrane.github.io/django/)
# Components:
## Sentence transformer
Converts textual content (like PDF text or chunks) into vector embeddings.

Enables semantic search, clustering, or similarity comparison between text pieces.

Used to connect related text nodes in the graph based on meaning rather than exact words.

## CLIP
Converts images (extracted from PDFs) into vector embeddings in the same space as text embeddings.

Allows multimodal similarity search between text and images.

Supports linking image nodes to text nodes in the graph based on semantic relevance.
## Neo4j Graph Database
Storing nodes and relationships:

Text chunks and associated images are converted into graph nodes.

Relationships are created to link nodes based on content, context, or source pages.

Each node and relationship can store metadata, such as source_chunk_id, for traceability.

Retrieving nodes:

Nodes can be queried via properties (e.g., name, productId) or traversals (e.g., BFS up to 3 hops).

Enables fetching related nodes or connected content efficiently for recommendations or further analysis.

Duplicate detection and merging:

Nodes with the same id are detected using a Cypher query and merged.

Relationships connected to duplicates are preserved via apoc.refactor.mergeNodes.

This ensures a clean graph without redundant nodes while maintaining all connections.

Integration with embeddings:

Stored nodes can be combined with text or image embeddings for semantic or multimodal search



## Langchain
Model Integration

Uses OllamaLLM (e.g., llama3.2:1b) as the underlying LLM for reasoning over product data.

Acts as the language interface to process user queries and generate responses.

Prompt Template

ChatPromptTemplate is used to define a structured prompt with placeholders:

reviews: list of product nodes fetched from Neo4j.

bfs: related nodes retrieved via DFS/BFS traversal.

question: natural language query from the user.

Ensures the model only uses provided node data and connections to answer.

Query Chain Execution

Combines the prompt with the LLM using chain = prompt | model1.

chain.invoke() takes the populated placeholders and returns a structured answer.

Designed for product recommendation: outputs a paragraph with product details (name, price, category, image path) without extra commentary.

Integration with Retrieval and Embeddings

LangChain leverages FAISS embeddings (CLIP-based) for semantic search over products.

Neo4j nodes and graph connections feed into the prompt, giving context-aware recommendations.
## FAISS
FAISS Usage

Purpose

FAISS is used for fast similarity search over product embeddings.

Combines text and image embeddings (from CLIP) to perform semantic search.

Embedding Index

Each product has a combined text + image embedding.

FAISS IndexFlatL2 is used for L2 distance-based search.

IndexIDMap maps embeddings to product IDs, enabling retrieval of product metadata.

Querying

User queries are converted to embeddings using the same CLIP text encoder.

index_with_ids.search(query_embedding, k) retrieves the top k most similar products.

Integration

Retrieved product IDs from FAISS are then used to fetch full product info from Neo4j.

Works together with graph traversal (getNodes) and LangChain to generate recommendation responses.
## LLM
Purpose

The LLM is used to generate natural-language answers for user queries based on structured product and graph data.

Input includes:

Product nodes (reviews) with details like name, category, price, image path

Related nodes or graph connections (bfs)

User’s natural-language question (question)
```bash
# Clone the repository
git clone https://github.com/your-username/project-name.git

# Navigate to the project directory
cd project-name

# Install dependencies
pip install -r requirements.txt
```

---

## How to Use

### 1. Environment Setup

Create a `.env` file in the project root with the following variables:

```env
GOOGLE_API_KEY=your_google_api_key
NEO4J_AURA_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_AURA_USERNAME=neo4j
NEO4J_AURA_PASSWORD=your_password
NEO4J_AURA_DATABASE=neo4j
```

### 2. CSV GraphRAG (`csv_rag.py`)

```bash
python csv_rag.py
# Opens at http://127.0.0.1:7861
```

**Steps:**
1. **Upload** any `.csv` file.
2. **Select node columns** — pick which CSV columns become graph nodes (e.g. `product_name`, `category`, `brand`).
3. **Define relationships** — add rows to the table specifying source column → target column → label (e.g. `product_name` → `category` → `BELONGS_TO`).
4. **Build Graph** — generates CLIP embeddings and inserts nodes/relationships into Neo4j.
5. **Query** — switch to the Query tab and ask natural-language questions.

**Image support (optional):**
- If your CSV has an `image_path` column, place the corresponding image files inside the `static/` folder.
- The code looks for images at `static/<image_path_value>` and uses CLIP to create a combined text + image embedding.
- If no `image_path` column exists, or images are missing, the pipeline uses **text-only** embeddings — everything still works.

**Example CSV format:**

```csv
product_id,product_name,category,brand,price,description,image_path
1,Wireless Mouse,Electronics,Logitech,899,Ergonomic wireless mouse,mouse.jpg
2,Running Shoes,Footwear,Nike,5999,Lightweight running shoes,shoes.jpg
```

> The `image_path` column is optional. Without it the pipeline runs in text-only mode.

### 3. PDF GraphRAG (`pdf_rag.py`)

```bash
python pdf_rag.py
# Opens at http://127.0.0.1:7860
```

1. Upload a **PDF** file — text and images are extracted automatically.
2. The pipeline builds a knowledge graph from the PDF content.
3. Query the graph from the **Query** tab.

