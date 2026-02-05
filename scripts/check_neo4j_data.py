from neo4j import GraphDatabase
import yaml

def load_config():
    with open('config/neo4j.yaml', 'r') as f:
        return yaml.safe_load(f)

def check_data():
    config = load_config()['neo4j']
    driver = GraphDatabase.driver(
        config['uri'],
        auth=(config['username'], config['password'])
    )

    with driver.session() as session:
        # Count nodes
        result = session.run("MATCH (n) RETURN count(n) as count")
        count = result.single()['count']
        print(f"Total nodes: {count}")
        
        if count == 0:
            print("No nodes found in database")
            return
        
        # Count relationships
        result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
        rel_count = result.single()['count']
        print(f"Total relationships: {rel_count}")

        # Node types
        result = session.run("MATCH (n) RETURN labels(n) as labels, count(*) as count")
        print("Node counts:")
        for record in result:
            labels = record['labels']
            count = record['count']
            if labels:
                label_str = ':'.join(labels)
                print(f"  {label_str}: {count}")
            else:
                print(f"  Unlabeled: {count}")

        # Sample data - get all nodes
        print("\nSample nodes:")
        result = session.run("MATCH (n) RETURN labels(n) as labels, properties(n) as props LIMIT 5")
        for record in result:
            labels = record['labels']
            props = record['props']
            print(f"  {labels}: {props}")

        # Count relationships
        result = session.run("MATCH ()-[r]->() RETURN type(r) as type, count(*) as count")
        print("\nRelationship counts:")
        for record in result:
            print(f"  {record['type']}: {record['count']}")

        # Sample data - get all nodes
        print("\nAll nodes and properties:")
        result = session.run("MATCH (n) RETURN labels(n) as labels, properties(n) as props LIMIT 10")
        for record in result:
            labels = record['labels']
            props = record['props']
            print(f"  {labels}: {props}")

    driver.close()

if __name__ == "__main__":
    check_data()