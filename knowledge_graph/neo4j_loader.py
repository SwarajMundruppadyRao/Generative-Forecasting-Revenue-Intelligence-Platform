"""
Neo4j knowledge graph loader
"""
from neo4j import GraphDatabase
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path

from utils.config import NEO4J_CONFIG, BASE_DIR, TRAIN_FILE, STORES_FILE
from utils.logger import setup_logger

logger = setup_logger(__name__, BASE_DIR / "logs" / "neo4j.log")


class Neo4jLoader:
    """Load data into Neo4j knowledge graph"""
    
    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None
    ):
        """
        Initialize Neo4j connection
        
        Args:
            uri: Neo4j URI
            user: Username
            password: Password
        """
        self.uri = uri or NEO4J_CONFIG['uri']
        self.user = user or NEO4J_CONFIG['user']
        self.password = password or NEO4J_CONFIG['password']
        
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            logger.info(f"Connected to Neo4j at {self.uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self.driver = None
    
    def close(self):
        """Close connection"""
        if self.driver:
            self.driver.close()
            logger.info("Closed Neo4j connection")
    
    def clear_database(self):
        """Clear all nodes and relationships"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Cleared database")
    
    def create_constraints(self):
        """Create constraints and indices"""
        with self.driver.session() as session:
            # Create constraints
            constraints = [
                "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Store) REQUIRE s.store_id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Department) REQUIRE d.dept_id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (i:Item) REQUIRE i.item_id IS UNIQUE"
            ]
            
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    logger.warning(f"Constraint creation warning: {e}")
            
            logger.info("Created constraints")
    
    def load_stores(self, stores_data: pd.DataFrame):
        """Load store nodes"""
        with self.driver.session() as session:
            for _, row in stores_data.iterrows():
                session.run(
                    """
                    MERGE (s:Store {store_id: $store_id})
                    SET s.type = $type,
                        s.size = $size
                    """,
                    store_id=int(row['Store']),
                    type=row.get('Type', 'Unknown'),
                    size=int(row.get('Size', 0))
                )
            
            logger.info(f"Loaded {len(stores_data)} stores")
    
    def load_departments(self, train_data: pd.DataFrame):
        """Load department nodes"""
        departments = train_data['Dept'].unique()
        
        with self.driver.session() as session:
            for dept in departments:
                dept_data = train_data[train_data['Dept'] == dept]
                
                session.run(
                    """
                    MERGE (d:Department {dept_id: $dept_id})
                    SET d.avg_sales = $avg_sales,
                        d.total_sales = $total_sales,
                        d.num_records = $num_records
                    """,
                    dept_id=int(dept),
                    avg_sales=float(dept_data['Weekly_Sales'].mean()),
                    total_sales=float(dept_data['Weekly_Sales'].sum()),
                    num_records=int(len(dept_data))
                )
            
            logger.info(f"Loaded {len(departments)} departments")
    
    def create_relationships(self, train_data: pd.DataFrame):
        """Create relationships between stores and departments"""
        with self.driver.session() as session:
            # Store -> Department relationships
            for _, row in train_data.groupby(['Store', 'Dept']).agg({
                'Weekly_Sales': ['mean', 'sum', 'count']
            }).reset_index().iterrows():
                
                store_id = int(row['Store'])
                dept_id = int(row['Dept'])
                avg_sales = float(row['Weekly_Sales']['mean'])
                total_sales = float(row['Weekly_Sales']['sum'])
                count = int(row['Weekly_Sales']['count'])
                
                session.run(
                    """
                    MATCH (s:Store {store_id: $store_id})
                    MATCH (d:Department {dept_id: $dept_id})
                    MERGE (s)-[r:SELLS]->(d)
                    SET r.avg_sales = $avg_sales,
                        r.total_sales = $total_sales,
                        r.num_transactions = $count
                    """,
                    store_id=store_id,
                    dept_id=dept_id,
                    avg_sales=avg_sales,
                    total_sales=total_sales,
                    count=count
                )
            
            logger.info("Created store-department relationships")
    
    def create_similarity_relationships(self, threshold: float = 0.7):
        """Create similarity relationships between stores"""
        with self.driver.session() as session:
            # Find similar stores based on type and size
            session.run(
                """
                MATCH (s1:Store), (s2:Store)
                WHERE s1.store_id < s2.store_id
                  AND s1.type = s2.type
                  AND abs(s1.size - s2.size) < s1.size * 0.3
                MERGE (s1)-[r:SIMILAR_TO]-(s2)
                SET r.similarity = 0.8
                """
            )
            
            logger.info("Created similarity relationships")
    
    def load_all_data(
        self,
        train_data: pd.DataFrame,
        stores_data: pd.DataFrame,
        clear_first: bool = True
    ):
        """
        Load all data into Neo4j
        
        Args:
            train_data: Training data
            stores_data: Store metadata
            clear_first: Clear database before loading
        """
        if clear_first:
            self.clear_database()
        
        self.create_constraints()
        self.load_stores(stores_data)
        self.load_departments(train_data)
        self.create_relationships(train_data)
        self.create_similarity_relationships()
        
        logger.info("Loaded all data into Neo4j")
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (n)
                RETURN labels(n) as label, count(n) as count
                """
            )
            
            node_counts = {record['label'][0]: record['count'] for record in result}
            
            result = session.run(
                """
                MATCH ()-[r]->()
                RETURN type(r) as type, count(r) as count
                """
            )
            
            rel_counts = {record['type']: record['count'] for record in result}
            
            return {
                'nodes': node_counts,
                'relationships': rel_counts
            }


def main():
    """Load data into Neo4j"""
    # Load data
    train_data = pd.read_csv(TRAIN_FILE)
    stores_data = pd.read_csv(STORES_FILE)
    
    # Sample data for faster loading (optional)
    train_data = train_data.sample(min(10000, len(train_data)), random_state=42)
    
    # Load into Neo4j
    loader = Neo4jLoader()
    
    if loader.driver:
        loader.load_all_data(train_data, stores_data)
        
        # Print stats
        stats = loader.get_stats()
        print("\nNeo4j Database Statistics:")
        print("Nodes:", stats['nodes'])
        print("Relationships:", stats['relationships'])
        
        loader.close()
    else:
        print("Failed to connect to Neo4j. Please ensure Neo4j is running.")


if __name__ == "__main__":
    main()
