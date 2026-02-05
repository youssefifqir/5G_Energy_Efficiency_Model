#!/usr/bin/env python3
"""
Simple Neo4j connection test
"""

from neo4j import GraphDatabase
import yaml

def main():
    # Load config
    with open('config/neo4j.yaml', 'r') as f:
        config = yaml.safe_load(f)['neo4j']

    print(f"Connecting to: {config['uri']}")
    print(f"Username: {config['username']}")

    # Test connection
    try:
        driver = GraphDatabase.driver(config['uri'], auth=(config['username'], config['password']))

        with driver.session() as session:
            result = session.run('RETURN "Neo4j connection successful!" as message')
            record = result.single()
            print(f"✅ {record['message']}")

            # Test schema creation
            session.run("CREATE CONSTRAINT test_constraint IF NOT EXISTS FOR (n:TestNode) REQUIRE n.id IS UNIQUE")
            print("✅ Schema creation works")

        driver.close()
        print("🎉 All tests passed!")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check:")
        print("1. Neo4j Desktop is running")
        print("2. Password in config/neo4j.yaml is correct")
        print("3. Database is started")

if __name__ == "__main__":
    main()