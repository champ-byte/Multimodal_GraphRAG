import pandas as pd
from neo4j import GraphDatabase
from graphdatascience import GraphDataScience

# Neo4j connection (driver)
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "testpassword"

# GDS client using official graphdatascience library
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
gds = GraphDataScience(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Optional: verify GDS version
print("GDS Version:", gds.server_version())

# Read CSVs
users = pd.read_csv("users.csv")
products = pd.read_csv("products.csv")
purchases = pd.read_csv("purchases.csv")
friendships = pd.read_csv("friendships.csv")

gds = GraphDataScience(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# Read CSVs
users = pd.read_csv("users.csv")
products = pd.read_csv("products.csv")
purchases = pd.read_csv("purchases.csv")
friendships = pd.read_csv("friendships.csv")


# Clear existing data
driver.execute_query("MATCH (n) DETACH DELETE n")

# Create Users
for idx, row in users.iterrows():
    driver.execute_query(
        """
        CREATE (:User {
            userId: $uid,
            name: $name,
            age: $age,
            location: $loc
        })
        """,
        uid=int(row['userId']),
        name=row['name'],
        age=int(row['age']),
        loc=row["location"]
    )

# Create Products
for idx, row in products.iterrows():
    driver.execute_query(
        """
        CREATE (:Product {
            productId: $pid,
            name: $name,
            category: $cat,
            price: $price
        })
        """,
        pid=int(row['productId']),
        name=row['name'],
        cat=row['category'],
        price=float(row['price'])
    )


# Create Purchases
for idx, row in purchases.iterrows():
    driver.execute_query(
        """
        MATCH (u:User {userId: $uid}), (p:Product {productId: $pid})
        CREATE (u)-[:PURCHASED]->(p)
        """,
        uid=int(row['userId']),
        pid=int(row['productId'])
    )

# Create Friendships

for idx, row in friendships.iterrows():
    driver.execute_query(
        """
        MATCH (u1:User {userId: $u1}), (u2:User {userId: $u2})
        MERGE (u1)-[:FRIEND]->(u2)
        MERGE (u2)-[:FRIEND]->(u1)
        """,
        u1=int(row['userId1']),
        u2=int(row['userId2'])
    )
gds.graph.drop("friendGraph", False)
gds.graph.drop("userProductGraph", False)

# Project a friendship graph
G_friend, result = gds.graph.project(
    "friendGraph",
    "User",
    {"FRIEND": {"orientation": "UNDIRECTED"}}
)
assert G_friend.node_count() == result["nodeCount"]
print(" Friendship graph projected:", G_friend)

# Compute PageRank on the friendship graph
pr = gds.pageRank.stream(G_friend)
print("\n🔹 Top 5 Influential Users (PageRank):")
print(pr[["nodeId", "score"]]
      .assign(user=lambda df: df.nodeId.map(lambda nid: gds.util.asNode(nid)["name"]))
      .sort_values("score", ascending=False)
      .head(5))
# Project User–Product Bipartite Graph
G_up, result = gds.graph.project(
    "userProductGraph",
    ["User", "Product"],
    {"PURCHASED": {"orientation": "UNDIRECTED"}}
)
# nodeSimilarity
records, summary, keys = driver.execute_query("""
CALL gds.nodeSimilarity.stream('userProductGraph')
YIELD node1, node2, similarity
WITH gds.util.asNode(node1) AS n1, gds.util.asNode(node2) AS n2, similarity
WHERE n1:User AND n2:User
RETURN n1.name AS user1, n2.name AS user2, similarity
ORDER BY similarity ASC
LIMIT 10
""")

print("\n🔹 Top User–User Similarities:")
for r in records:
    print(f"{r['user1']} ↔ {r['user2']} | similarity: {r['similarity']:.3f}")
# Shortest Path between Alice and Noah
records, summary, keys = driver.execute_query("""
MATCH (source:User {name:'Alice'}), (target:User {name:'Noah'})
CALL gds.shortestPath.dijkstra.stream('friendGraph', {
  sourceNode: id(source),
  targetNode: id(target)
})
YIELD totalCost, nodeIds
RETURN totalCost,
       [n IN gds.util.asNodes(nodeIds) | n.name] AS path
""")

print("\n🔹 Shortest Path (Alice → Noah):")
for r in records:
    print(f"Path: {' -> '.join(r['path'])} | Cost: {r['totalCost']}")

#Degree Centrality
records, summary, keys = driver.execute_query("""
CALL gds.degree.stream('friendGraph')
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).name AS user, score AS degree
ORDER BY degree DESC
LIMIT 4
""")

print("\n🔹 Degree Centrality Top 4:")
for r in records:
    print(f"{r['user']} | degree: {r['degree']}")
