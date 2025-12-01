"""
Neo4j query utilities
"""
from neo4j import GraphDatabase
from typing import Dict, List, Optional, Any
import pandas as pd

from utils.config import NEO4J_CONFIG
from utils.logger import get_logger

logger = get_logger(__name__)


class Neo4jQuery:
    """Query Neo4j knowledge graph"""
    
    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None
    ):
        """Initialize Neo4j connection"""
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
    
    def execute_query(self, query: str, parameters: Optional[Dict] = None) -> List[Dict]:
        """Execute Cypher query"""
        if not self.driver:
            return []
        
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [dict(record) for record in result]
    
    def get_store_info(self, store_id: int) -> Dict:
        """Get store information"""
        query = """
        MATCH (s:Store {store_id: $store_id})
        RETURN s.store_id as store_id, s.type as type, s.size as size
        """
        
        results = self.execute_query(query, {'store_id': store_id})
        return results[0] if results else {}
    
    def get_store_departments(self, store_id: int) -> List[Dict]:
        """Get departments sold by a store"""
        query = """
        MATCH (s:Store {store_id: $store_id})-[r:SELLS]->(d:Department)
        RETURN d.dept_id as dept_id,
               r.avg_sales as avg_sales,
               r.total_sales as total_sales,
               r.num_transactions as num_transactions
        ORDER BY r.total_sales DESC
        """
        
        return self.execute_query(query, {'store_id': store_id})
    
    def get_similar_stores(self, store_id: int, limit: int = 5) -> List[Dict]:
        """Get similar stores"""
        query = """
        MATCH (s1:Store {store_id: $store_id})-[r:SIMILAR_TO]-(s2:Store)
        RETURN s2.store_id as store_id,
               s2.type as type,
               s2.size as size,
               r.similarity as similarity
        ORDER BY r.similarity DESC
        LIMIT $limit
        """
        
        return self.execute_query(query, {'store_id': store_id, 'limit': limit})
    
    def get_top_departments(self, limit: int = 10) -> List[Dict]:
        """Get top departments by total sales"""
        query = """
        MATCH (d:Department)
        RETURN d.dept_id as dept_id,
               d.avg_sales as avg_sales,
               d.total_sales as total_sales,
               d.num_records as num_records
        ORDER BY d.total_sales DESC
        LIMIT $limit
        """
        
        return self.execute_query(query, {'limit': limit})
    
    def get_department_stores(self, dept_id: int) -> List[Dict]:
        """Get stores selling a department"""
        query = """
        MATCH (s:Store)-[r:SELLS]->(d:Department {dept_id: $dept_id})
        RETURN s.store_id as store_id,
               s.type as type,
               s.size as size,
               r.avg_sales as avg_sales,
               r.total_sales as total_sales
        ORDER BY r.total_sales DESC
        """
        
        return self.execute_query(query, {'dept_id': dept_id})
    
    def get_store_insights(self, store_id: int) -> Dict:
        """
        Get comprehensive insights for a store
        
        Returns:
            Dictionary with store info, departments, and similar stores
        """
        insights = {
            'store_info': self.get_store_info(store_id),
            'departments': self.get_store_departments(store_id),
            'similar_stores': self.get_similar_stores(store_id)
        }
        
        return insights
    
    def find_stores_by_criteria(
        self,
        store_type: Optional[str] = None,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None
    ) -> List[Dict]:
        """Find stores matching criteria"""
        conditions = []
        params = {}
        
        if store_type:
            conditions.append("s.type = $store_type")
            params['store_type'] = store_type
        
        if min_size:
            conditions.append("s.size >= $min_size")
            params['min_size'] = min_size
        
        if max_size:
            conditions.append("s.size <= $max_size")
            params['max_size'] = max_size
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        query = f"""
        MATCH (s:Store)
        WHERE {where_clause}
        RETURN s.store_id as store_id, s.type as type, s.size as size
        ORDER BY s.size DESC
        """
        
        return self.execute_query(query, params)
    
    def get_revenue_context(self, store_id: int, dept_id: Optional[int] = None) -> str:
        """
        Get textual context about revenue for RAG
        
        Args:
            store_id: Store ID
            dept_id: Optional department ID
        
        Returns:
            Textual description
        """
        insights = self.get_store_insights(store_id)
        
        context = f"Store {store_id} is a "
        
        if insights['store_info']:
            info = insights['store_info']
            context += f"type {info.get('type', 'Unknown')} store with size {info.get('size', 0)} sq ft. "
        
        if insights['departments']:
            top_depts = insights['departments'][:3]
            context += f"Top departments: "
            for dept in top_depts:
                context += f"Dept {dept['dept_id']} (avg sales: ${dept['avg_sales']:.2f}), "
        
        if insights['similar_stores']:
            similar = insights['similar_stores'][:2]
            context += f"Similar stores: "
            for store in similar:
                context += f"Store {store['store_id']} (type {store['type']}), "
        
        return context.strip()


def main():
    """Test Neo4j queries"""
    query = Neo4jQuery()
    
    if query.driver:
        # Test queries
        print("\nTop 5 Departments:")
        top_depts = query.get_top_departments(5)
        for dept in top_depts:
            print(f"  Dept {dept['dept_id']}: ${dept['total_sales']:.2f} total sales")
        
        print("\nStore 1 Insights:")
        insights = query.get_store_insights(1)
        print(f"  Info: {insights['store_info']}")
        print(f"  Departments: {len(insights['departments'])}")
        print(f"  Similar stores: {len(insights['similar_stores'])}")
        
        print("\nRevenue Context for Store 1:")
        context = query.get_revenue_context(1)
        print(f"  {context}")
        
        query.close()
    else:
        print("Failed to connect to Neo4j")


if __name__ == "__main__":
    main()
