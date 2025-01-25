# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class BipartiteSAGEConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BipartiteSAGEConv, self).__init__()
        self.sage_conv = SAGEConv(in_channels, out_channels, normalize=True)
        
    def forward(self, x, edge_index, size=None):
        return self.sage_conv(x, edge_index, size=size)

class BM3Layer(nn.Module):
    def __init__(self, hidden_dim, num_nodes_wm, num_nodes_gm, dropout=0.0):
        super(BM3Layer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes_wm = num_nodes_wm
        self.num_nodes_gm = num_nodes_gm
        
        self.wm_transform = nn.Linear(num_nodes_wm, hidden_dim)
        self.gm_transform = nn.Linear(num_nodes_gm, hidden_dim)
        
        self.sage_wm_to_gm_1 = BipartiteSAGEConv(hidden_dim, hidden_dim)
        self.sage_wm_to_gm_2 = BipartiteSAGEConv(hidden_dim, hidden_dim)
        
        self.sage_gm_to_wm_1 = BipartiteSAGEConv(hidden_dim, hidden_dim)
        self.sage_gm_to_wm_2 = BipartiteSAGEConv(hidden_dim, hidden_dim)
        
        self.wm_update = nn.Linear(hidden_dim * 3, hidden_dim)
        self.gm_update = nn.Linear(hidden_dim * 3, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm_wm = nn.LayerNorm(hidden_dim)
        self.layer_norm_gm = nn.LayerNorm(hidden_dim)
        
    def _prepare_graph_input(self, connectivity):
        batch_size = connectivity.size(0)
        device = connectivity.device
        all_edges_src = []
        all_edges_dst = []
        
        for b in range(batch_size):
            conn = connectivity[b]
            rows, cols = torch.where(conn > 0)

            wm_offset = b * self.num_nodes_wm
            gm_offset = b * self.num_nodes_gm
            
            src = rows + wm_offset
            dst = cols + gm_offset
            
            all_edges_src.append(src)
            all_edges_dst.append(dst)
        
        edge_index_src = torch.cat(all_edges_src)
        edge_index_dst = torch.cat(all_edges_dst)
        
        edge_index = torch.stack([edge_index_src, edge_index_dst])
        
        return edge_index.to(device)

    def forward(self, wm_features, gm_features, connectivity, i):
        batch_size = wm_features.size(0)
        device = wm_features.device
        
        wm_features = wm_features.to(torch.float32).to(device)
        gm_features = gm_features.to(torch.float32).to(device)
        connectivity = connectivity.to(torch.float32).to(device)
        
        wm_h = self.wm_transform(wm_features) if i==0 else wm_features 
        gm_h = self.gm_transform(gm_features) if i==0 else gm_features
        
        wm_h_flat = wm_h.reshape(-1, self.hidden_dim)
        gm_h_flat = gm_h.reshape(-1, self.hidden_dim)
        
        x = torch.cat([wm_h_flat, gm_h_flat], dim=0)
        edge_index = self._prepare_graph_input(connectivity)
        
        total_nodes_per_batch = self.num_nodes_wm + self.num_nodes_gm
        total_nodes = total_nodes_per_batch * batch_size
        
        wm_to_gm_1 = F.relu(self.sage_wm_to_gm_1(x, edge_index))
        wm_to_gm_2 = F.relu(self.sage_wm_to_gm_2(wm_to_gm_1, edge_index))
        
        edge_index_reverse = torch.stack([edge_index[1], edge_index[0]], dim=0)
        gm_to_wm_1 = F.relu(self.sage_gm_to_wm_1(x, edge_index_reverse))
        gm_to_wm_2 = F.relu(self.sage_gm_to_wm_2(gm_to_wm_1, edge_index_reverse))
        
        wm_msg = wm_to_gm_2[:batch_size * self.num_nodes_wm].view(batch_size, self.num_nodes_wm, -1)
        gm_msg = gm_to_wm_2[batch_size * self.num_nodes_wm:].view(batch_size, self.num_nodes_gm, -1)
        
        wm_combined = torch.cat([wm_h, wm_msg, torch.bmm(connectivity, gm_h)], dim=-1)
        gm_combined = torch.cat([gm_h, gm_msg, torch.bmm(connectivity.transpose(1, 2), wm_h)], dim=-1)
        
        wm_out = self.wm_update(wm_combined)
        wm_out = F.relu(self.layer_norm_wm(wm_out))
        wm_out = self.dropout(wm_out)
        
        gm_out = self.gm_update(gm_combined)
        gm_out = F.relu(self.layer_norm_gm(gm_out))
        gm_out = self.dropout(gm_out)
        
        return wm_out, gm_out

class BiMatterAdversary(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(BiMatterAdversary, self).__init__()
        self.feature_transform = nn.Linear(input_dim, hidden_dim)
        self.discriminator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        

    def forward(self, original_features, learned_features):
        batch_size = original_features.size(0)
        num_nodes = original_features.size(1)

        original_transformed = self.feature_transform(original_features) 
        combined = torch.cat([original_transformed, learned_features], dim=-1) 
        combined = combined.view(-1, combined.size(-1))  
        
        discriminator_output = self.discriminator(combined) 
        
        return discriminator_output.view(batch_size, num_nodes, 1)


class BrainGraphClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_nodes_wm=48, num_nodes_gm=18, layers=2, dropout=0.0):
        super(BrainGraphClassifier, self).__init__()
        
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.num_nodes_wm = num_nodes_wm
        self.num_nodes_gm = num_nodes_gm
        self.last_gm_embeddings = []
        
        self.bm3_layers = nn.ModuleList([
            BM3Layer(
                hidden_dim=hidden_dim,
                num_nodes_wm=num_nodes_wm,
                num_nodes_gm=num_nodes_gm,
                dropout=dropout,
            ) for i in range(layers)
        ])
        
        self.adversary_wm = nn.ModuleList([
            BiMatterAdversary(num_nodes_wm, hidden_dim)
            for _ in range(layers)
        ])
        self.adversary_gm = nn.ModuleList([
            BiMatterAdversary(num_nodes_gm, hidden_dim)
            for _ in range(layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )

    def get_gm_embeddings(self):
        return self.last_gm_embeddings 

    def forward(self, data):
        wm_features = data['wm_time']  
        gm_features = data['gm_time'] 
        connectivity = data['connectivity'] 

        wm_features = wm_features.to(torch.float32)
        gm_features = gm_features.to(torch.float32)
        connectivity = connectivity.to(torch.float32)
        
        batch_size = wm_features.size(0)
        adv_losses = []

        wm_original = wm_features
        gm_original = gm_features

        for i in range(self.layers):
            wm_features, gm_features = self.bm3_layers[i](
                wm_features, gm_features, connectivity, i
            )
            wm_adv_loss = self.adversary_wm[i](wm_original, wm_features).mean()
            gm_adv_loss = self.adversary_gm[i](gm_original, gm_features).mean()
            adv_losses.append(wm_adv_loss + gm_adv_loss)

        wm_graph = wm_features.mean(dim=1) 
        gm_graph = gm_features.mean(dim=1) 
        graph_repr = torch.cat([wm_graph, gm_graph], dim=1) 
        logits = self.classifier(graph_repr)
        
        return logits, sum(adv_losses) / len(adv_losses) if adv_losses else 0, wm_features, gm_features
