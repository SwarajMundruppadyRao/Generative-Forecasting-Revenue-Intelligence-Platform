"""
GNN model for revenue influence prediction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv, HeteroConv, Linear
from torch_geometric.data import HeteroData
from typing import Dict, Optional

from utils.config import GNN_CONFIG
from utils.logger import get_logger

logger = get_logger(__name__)


class HeteroGNN(nn.Module):
    """Heterogeneous GNN for revenue prediction"""
    
    def __init__(
        self,
        metadata: tuple,
        hidden_channels: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3,
        heads: int = 4,
        model_type: str = "GAT"
    ):
        """
        Initialize Heterogeneous GNN
        
        Args:
            metadata: Graph metadata (node_types, edge_types)
            hidden_channels: Hidden dimension
            num_layers: Number of GNN layers
            dropout: Dropout rate
            heads: Number of attention heads (for GAT)
            model_type: 'GAT' or 'GraphSAGE'
        """
        super(HeteroGNN, self).__init__()
        
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.model_type = model_type
        
        # Create heterogeneous convolutions
        self.convs = nn.ModuleList()
        
        for i in range(num_layers):
            conv_dict = {}
            
            for edge_type in metadata[1]:
                src_type, _, dst_type = edge_type
                
                if model_type == "GAT":
                    if i == 0:
                        conv_dict[edge_type] = GATConv(
                            (-1, -1),
                            hidden_channels // heads,
                            heads=heads,
                            dropout=dropout,
                            add_self_loops=False
                        )
                    else:
                        conv_dict[edge_type] = GATConv(
                            (hidden_channels, hidden_channels),
                            hidden_channels // heads,
                            heads=heads,
                            dropout=dropout,
                            add_self_loops=False
                        )
                else:  # GraphSAGE
                    conv_dict[edge_type] = SAGEConv(
                        (-1, -1),
                        hidden_channels,
                        add_self_loops=False
                    )
            
            self.convs.append(HeteroConv(conv_dict, aggr='mean'))
        
        # Output layers for different tasks
        self.revenue_predictor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1)
        )
        
        self.similarity_predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[tuple, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x_dict: Node features dictionary
            edge_index_dict: Edge indices dictionary
        
        Returns:
            Node embeddings dictionary
        """
        # Apply GNN layers
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            
            # Apply activation and dropout
            x_dict = {
                key: F.relu(x) if i < self.num_layers - 1 else x
                for key, x in x_dict.items()
            }
            
            if i < self.num_layers - 1:
                x_dict = {
                    key: F.dropout(x, p=self.dropout, training=self.training)
                    for key, x in x_dict.items()
                }
        
        return x_dict
    
    def predict_revenue_influence(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[tuple, torch.Tensor],
        node_type: str = 'dept'
    ) -> torch.Tensor:
        """
        Predict revenue influence scores
        
        Args:
            x_dict: Node features
            edge_index_dict: Edge indices
            node_type: Node type to predict for
        
        Returns:
            Revenue influence scores
        """
        embeddings = self.forward(x_dict, edge_index_dict)
        
        if node_type in embeddings:
            scores = self.revenue_predictor(embeddings[node_type])
            return scores
        else:
            raise ValueError(f"Node type {node_type} not found in embeddings")
    
    def predict_similarity(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[tuple, torch.Tensor],
        node_pairs: torch.Tensor,
        node_type: str = 'store'
    ) -> torch.Tensor:
        """
        Predict similarity between node pairs
        
        Args:
            x_dict: Node features
            edge_index_dict: Edge indices
            node_pairs: Tensor of shape (2, num_pairs)
            node_type: Node type
        
        Returns:
            Similarity scores
        """
        embeddings = self.forward(x_dict, edge_index_dict)
        
        if node_type in embeddings:
            node_emb = embeddings[node_type]
            
            # Get embeddings for pairs
            src_emb = node_emb[node_pairs[0]]
            dst_emb = node_emb[node_pairs[1]]
            
            # Concatenate and predict
            pair_emb = torch.cat([src_emb, dst_emb], dim=1)
            scores = self.similarity_predictor(pair_emb)
            
            return scores
        else:
            raise ValueError(f"Node type {node_type} not found in embeddings")
    
    def get_embeddings(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[tuple, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Get node embeddings"""
        return self.forward(x_dict, edge_index_dict)


class GNNTrainer:
    """Trainer for GNN models"""
    
    def __init__(
        self,
        model: HeteroGNN,
        device: str = 'cpu'
    ):
        self.model = model
        self.device = device
        self.model.to(device)
        
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.001,
            weight_decay=5e-4
        )
    
    def train_epoch(
        self,
        graph: HeteroData,
        target_node_type: str = 'dept',
        target_values: Optional[torch.Tensor] = None
    ) -> float:
        """Train for one epoch"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move graph to device
        graph = graph.to(self.device)
        
        # Forward pass
        predictions = self.model.predict_revenue_influence(
            graph.x_dict,
            graph.edge_index_dict,
            node_type=target_node_type
        )
        
        # Compute loss
        if target_values is None:
            # Use dummy targets for unsupervised learning
            target_values = torch.zeros_like(predictions)
        
        target_values = target_values.to(self.device)
        loss = F.mse_loss(predictions, target_values)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def save_model(self, path):
        """Save model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
        logger.info(f"Saved GNN model to {path}")
    
    def load_model(self, path):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Loaded GNN model from {path}")


def create_gnn_model(metadata: tuple, config: dict = None) -> HeteroGNN:
    """
    Create GNN model
    
    Args:
        metadata: Graph metadata
        config: Model configuration
    
    Returns:
        GNN model
    """
    if config is None:
        config = GNN_CONFIG.copy()
    
    model = HeteroGNN(metadata, **config)
    
    return model


def main():
    """Test GNN model"""
    from graph.build_graph import GraphBuilder
    from utils.config import BASE_DIR
    import pandas as pd
    
    # Load graph
    builder = GraphBuilder()
    graph = builder.load_graph(BASE_DIR / 'graph' / 'sales_graph.pt')
    
    # Create model
    model = create_gnn_model(graph.metadata())
    
    logger.info(f"Created GNN model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    embeddings = model.get_embeddings(graph.x_dict, graph.edge_index_dict)
    
    for node_type, emb in embeddings.items():
        logger.info(f"{node_type} embeddings: {emb.shape}")


if __name__ == "__main__":
    main()
