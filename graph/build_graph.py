"""
Build graph structure from data
"""
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, HeteroData
from typing import Dict, List, Tuple, Optional
import pickle
from pathlib import Path

from utils.config import BASE_DIR, TRAIN_FILE, STORES_FILE
from utils.embedding import EmbeddingGenerator, batch_cosine_similarity
from utils.logger import setup_logger

logger = setup_logger(__name__, BASE_DIR / "logs" / "graph.log")


class GraphBuilder:
    """Build graph structure for GNN"""
    
    def __init__(self):
        self.embedding_generator = EmbeddingGenerator()
        self.node_mappings = {}
        self.edge_indices = {}
        
    def build_from_data(
        self,
        train_data: pd.DataFrame,
        stores_data: pd.DataFrame,
        similarity_threshold: float = 0.7
    ) -> HeteroData:
        """
        Build heterogeneous graph from data
        
        Args:
            train_data: Training data
            stores_data: Store metadata
            similarity_threshold: Threshold for item similarity edges
        
        Returns:
            HeteroData graph
        """
        logger.info("Building graph structure...")
        
        graph = HeteroData()
        
        # Create node mappings
        stores = train_data['Store'].unique()
        depts = train_data['Dept'].unique() if 'Dept' in train_data.columns else []
        
        self.node_mappings['store'] = {store: idx for idx, store in enumerate(stores)}
        self.node_mappings['dept'] = {dept: idx for idx, dept in enumerate(depts)}
        
        logger.info(f"Nodes - Stores: {len(stores)}, Departments: {len(depts)}")
        
        # Create store node features
        store_features = self._create_store_features(stores_data, stores)
        graph['store'].x = torch.FloatTensor(store_features)
        graph['store'].node_id = torch.LongTensor(list(stores))
        
        # Create department node features
        if len(depts) > 0:
            dept_features = self._create_dept_features(train_data, depts)
            graph['dept'].x = torch.FloatTensor(dept_features)
            graph['dept'].node_id = torch.LongTensor(list(depts))
        
        # Create edges: store -> dept
        if len(depts) > 0:
            store_dept_edges = self._create_store_dept_edges(train_data)
            graph['store', 'sells', 'dept'].edge_index = torch.LongTensor(store_dept_edges)
            
            # Reverse edges
            graph['dept', 'sold_by', 'store'].edge_index = torch.LongTensor(
                [store_dept_edges[1], store_dept_edges[0]]
            )
        
        # Create similarity edges between departments
        if len(depts) > 0:
            dept_similarity_edges = self._create_similarity_edges(
                dept_features,
                threshold=similarity_threshold
            )
            graph['dept', 'similar_to', 'dept'].edge_index = torch.LongTensor(dept_similarity_edges)
        
        # Create store similarity edges
        store_similarity_edges = self._create_similarity_edges(
            store_features,
            threshold=similarity_threshold
        )
        graph['store', 'similar_to', 'store'].edge_index = torch.LongTensor(store_similarity_edges)
        
        # Add edge attributes (revenue influence)
        if len(depts) > 0:
            edge_attrs = self._create_edge_attributes(train_data, store_dept_edges)
            graph['store', 'sells', 'dept'].edge_attr = torch.FloatTensor(edge_attrs)
        
        logger.info("Graph construction complete!")
        self._print_graph_stats(graph)
        
        return graph
    
    def _create_store_features(
        self,
        stores_data: pd.DataFrame,
        stores: np.ndarray
    ) -> np.ndarray:
        """Create store node features"""
        features = []
        
        for store in stores:
            store_info = stores_data[stores_data['Store'] == store]
            
            if len(store_info) == 0:
                # Default features
                feat = np.zeros(5)
            else:
                store_info = store_info.iloc[0]
                
                # Encode store type
                type_encoding = {'A': 0, 'B': 1, 'C': 2}
                store_type = type_encoding.get(store_info.get('Type', 'A'), 0)
                
                # Features: [type, size, normalized_size]
                size = store_info.get('Size', 0)
                feat = np.array([
                    store_type,
                    size,
                    size / 250000,  # Normalized size
                    store,  # Store ID as feature
                    np.log1p(size)  # Log size
                ])
            
            features.append(feat)
        
        return np.array(features)
    
    def _create_dept_features(
        self,
        train_data: pd.DataFrame,
        depts: np.ndarray
    ) -> np.ndarray:
        """Create department node features"""
        features = []
        
        for dept in depts:
            dept_data = train_data[train_data['Dept'] == dept]
            
            if len(dept_data) == 0:
                feat = np.zeros(6)
            else:
                # Aggregate statistics
                avg_sales = dept_data['Weekly_Sales'].mean()
                std_sales = dept_data['Weekly_Sales'].std()
                max_sales = dept_data['Weekly_Sales'].max()
                min_sales = dept_data['Weekly_Sales'].min()
                
                feat = np.array([
                    dept,  # Dept ID
                    avg_sales,
                    std_sales,
                    max_sales,
                    min_sales,
                    len(dept_data)  # Number of records
                ])
            
            features.append(feat)
        
        # Normalize
        features = np.array(features)
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
        
        return features
    
    def _create_store_dept_edges(
        self,
        train_data: pd.DataFrame
    ) -> np.ndarray:
        """Create edges between stores and departments"""
        edges = []
        
        for _, row in train_data.groupby(['Store', 'Dept']).size().reset_index().iterrows():
            store = row['Store']
            dept = row['Dept']
            
            store_idx = self.node_mappings['store'][store]
            dept_idx = self.node_mappings['dept'][dept]
            
            edges.append([store_idx, dept_idx])
        
        edges = np.array(edges).T
        logger.info(f"Created {edges.shape[1]} store-department edges")
        
        return edges
    
    def _create_similarity_edges(
        self,
        features: np.ndarray,
        threshold: float = 0.7
    ) -> np.ndarray:
        """Create similarity edges based on feature similarity"""
        # Compute cosine similarity
        similarity = batch_cosine_similarity(features, features)
        
        # Create edges for similar nodes
        edges = []
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                if similarity[i, j] > threshold:
                    edges.append([i, j])
                    edges.append([j, i])  # Undirected
        
        if len(edges) == 0:
            # Add self-loops if no edges
            edges = [[i, i] for i in range(len(features))]
        
        edges = np.array(edges).T
        logger.info(f"Created {edges.shape[1]} similarity edges")
        
        return edges
    
    def _create_edge_attributes(
        self,
        train_data: pd.DataFrame,
        edges: np.ndarray
    ) -> np.ndarray:
        """Create edge attributes (revenue influence)"""
        edge_attrs = []
        
        for i in range(edges.shape[1]):
            store_idx = edges[0, i]
            dept_idx = edges[1, i]
            
            # Get original IDs
            store = list(self.node_mappings['store'].keys())[store_idx]
            dept = list(self.node_mappings['dept'].keys())[dept_idx]
            
            # Get sales data
            sales_data = train_data[
                (train_data['Store'] == store) & (train_data['Dept'] == dept)
            ]['Weekly_Sales']
            
            if len(sales_data) > 0:
                # Revenue influence features
                attr = np.array([
                    sales_data.mean(),
                    sales_data.std(),
                    sales_data.sum()
                ])
            else:
                attr = np.zeros(3)
            
            edge_attrs.append(attr)
        
        edge_attrs = np.array(edge_attrs)
        
        # Normalize
        edge_attrs = (edge_attrs - edge_attrs.mean(axis=0)) / (edge_attrs.std(axis=0) + 1e-8)
        
        return edge_attrs
    
    def _print_graph_stats(self, graph: HeteroData):
        """Print graph statistics"""
        logger.info("\nGraph Statistics:")
        for node_type in graph.node_types:
            logger.info(f"  {node_type}: {graph[node_type].x.shape[0]} nodes, "
                       f"{graph[node_type].x.shape[1]} features")
        
        for edge_type in graph.edge_types:
            logger.info(f"  {edge_type}: {graph[edge_type].edge_index.shape[1]} edges")
    
    def save_graph(self, graph: HeteroData, path: Path):
        """Save graph to file"""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'graph': graph,
            'node_mappings': self.node_mappings
        }, path)
        
        logger.info(f"Saved graph to {path}")
    
    def load_graph(self, path: Path) -> HeteroData:
        """Load graph from file"""
        data = torch.load(path, weights_only=False)
        self.node_mappings = data['node_mappings']
        logger.info(f"Loaded graph from {path}")
        return data['graph']


def main():
    """Build and save graph"""
    # Load data
    train_data = pd.read_csv(BASE_DIR / 'data' / 'processed' / 'train_processed.csv')
    stores_data = pd.read_csv(STORES_FILE)
    
    # Build graph
    builder = GraphBuilder()
    graph = builder.build_from_data(train_data, stores_data)
    
    # Save graph
    builder.save_graph(graph, BASE_DIR / 'graph' / 'sales_graph.pt')
    
    print("Graph built and saved successfully!")


if __name__ == "__main__":
    main()
