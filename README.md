# Multimodal_GraphRAG


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
