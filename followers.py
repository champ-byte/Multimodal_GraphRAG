from neo4j import GraphDatabase
import pandas as pd

# URI examples: "neo4j://localhost", "neo4j+s://xxx.databases.neo4j.io"
URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "testpassword")

df=pd.read_csv('friendlist.csv')

with GraphDatabase.driver(URI, auth=AUTH) as driver:
    
    driver.verify_connectivity()
    driver.execute_query("MATCH (n) DETACH DELETE n", database_="neo4j")
    for index, row in df.iterrows():
        summary = driver.execute_query("""
            MERGE (a:Person {name: $name})
            MERGE (b:Person {name: $friendName})
            MERGE (a)-[:KNOWS]->(b)
            """,
            name=row['Person'], friendName=row['Friend'],
            database_="neo4j",
        ).summary
    print("Created {nodes_created} nodes in {time} ms.".format(
        nodes_created=summary.counters.nodes_created,
        time=summary.result_available_after
    ))
    records, summary, keys = driver.execute_query("""
    MATCH (p:Person)-[:KNOWS]->(f:Person)
    RETURN p.name AS name , f.name AS friend
    """,
    database_="neo4j",
    )
    for record in records:
        print(record.data())