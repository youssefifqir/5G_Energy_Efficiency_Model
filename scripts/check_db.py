import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_PATH)

from utils.neo4j_logger import Neo4jLogger

def check_database():
    neo4j_logger = Neo4jLogger()
    stats = neo4j_logger.get_database_stats()
    
    print("Database Statistics:")
    print("Nodes:")
    for label, count in stats['nodes'].items():
        print(f"  {label}: {count}")
    
    print("Relationships:")
    for rel_type, count in stats['relationships'].items():
        print(f"  {rel_type}: {count}")
    
    neo4j_logger.close()

if __name__ == "__main__":
    check_database()