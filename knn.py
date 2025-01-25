# knn.py
import torch
import torch.nn as nn
import numpy as np

class KNNModule:
    def __init__(self, k=5, num_nodes_wm=48, num_nodes_gm=18,device = "cuda:0"):
        self.k = k
        self.num_nodes_wm = num_nodes_wm
        self.num_nodes_gm = num_nodes_gm
        self.device = device

    def compute_semantic_similarity_batch(embeddings, num_nodes_gm):
        semantic_similarity_batch = []
        shape = embeddings.shape[0]
        embeddings = embeddings.reshape(embeddings.shape[0] // num_nodes_gm, num_nodes_gm, -1)
        for b in range(shape // num_nodes_gm):
            embeddings_batch = embeddings[b]
            norm_embeddings = embeddings_batch / torch.norm(embeddings_batch, dim=1, keepdim=True)
            semantic_similarity_batch.append(torch.mm(norm_embeddings, norm_embeddings.t()))
        return torch.stack(semantic_similarity_batch).to(embeddings.device)

    def normalize(matrix):
        if isinstance(matrix, torch.Tensor):
            matrix = matrix.detach().cpu().numpy() 
        min_val = np.min(matrix)
        max_val = np.max(matrix)
        return (matrix - min_val) / (max_val - min_val)
    
    def construct_knn_graph_batch(self,aa_index_batch, semantic_similarity_batch, k, num_wm_nodes, num_gm_nodes):
        knn_graph_batch = []
        for b in range(aa_index_batch.shape[0]):
            aa_index_normalized = self.normalize(aa_index_batch[b])
            semantic_similarity_normalized = self.normalize(semantic_similarity_batch[b].detach().cpu().numpy())
            
            combined_similarity = aa_index_normalized * semantic_similarity_normalized
            
            knn_graph = np.zeros_like(combined_similarity) 
            
            for i in range(combined_similarity.shape[0]):
                top_k_neighbors = np.argsort(combined_similarity[i])[-k:] 
                knn_graph[i, top_k_neighbors] = combined_similarity[i, top_k_neighbors]
            
            knn_graph_batch.append(knn_graph)
        knn_graph_batch_np = np.stack(knn_graph_batch)
        return torch.tensor(knn_graph_batch_np).to(aa_index_batch.device)

    
    def calculate_loss_with_knn_batch(u_embeddings, knn_graph_batch, num_gm_nodes):
        loss_u = 0
        
        u_embeddings_normalized = u_embeddings / torch.norm(u_embeddings, dim=1, keepdim=True)
        for b in range(knn_graph_batch.shape[0]):
            k_neighbors = knn_graph_batch[b]

            u_batch_embeddings = u_embeddings_normalized[b * num_gm_nodes:(b+1) * num_gm_nodes]
            u_i = u_batch_embeddings.unsqueeze(1) 
            u_j = u_batch_embeddings.unsqueeze(0) 
            
            similarity_matrix = torch.einsum('bik,bjk->bij', u_i, u_j)
            weighted_similarity = k_neighbors * similarity_matrix
            loss_u += weighted_similarity.sum()

        total_pairs = knn_graph_batch.sum()
        return loss_u / total_pairs if total_pairs > 0 else 0

    def forward(self, gm_embeddings, aa_index, alpha=0.5):
        aa_index_batch = aa_index.reshape(aa_index.shape[1] // self.num_nodes_gm, self.num_nodes_gm, -1).to(self.device)
        semantic_similarity_batch =self.compute_semantic_similarity_batch(gm_embeddings, self.num_nodes_gm)
        knn_graph_batch = self.construct_knn_graph_batch(aa_index_batch, semantic_similarity_batch, self.k, self.num_nodes_wm, self.num_nodes_gm).to(self.device)
        knn_loss = self.calculate_loss_with_knn_batch(gm_embeddings, knn_graph_batch, self.num_nodes_gm)
        return knn_loss