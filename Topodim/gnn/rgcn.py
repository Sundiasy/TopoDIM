import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch.distributions import Categorical

def _get_reachable_nodes(adj_bool, start_nodes):
    N = adj_bool.size(0)
    visited = torch.zeros(N, dtype=torch.bool, device=adj_bool.device)
    queue = list(start_nodes)
    for node in start_nodes:
        visited[node] = True
    
    q_ptr = 0
    while q_ptr < len(queue):
        u = queue[q_ptr]
        q_ptr += 1
        

        out_nbrs = torch.where(adj_bool[u])[0]
        in_nbrs = torch.where(adj_bool[:, u])[0]
        all_nbrs = torch.cat([out_nbrs, in_nbrs]).unique()
        
        for v in all_nbrs.tolist():
            if not visited[v]:
                visited[v] = True
                queue.append(v)
    
    return visited

def _is_reachable(adj_bool, src: int, tgt: int) -> bool:
    if src == tgt:
        return False
    N = adj_bool.size(0)
    visited = torch.zeros(N, dtype=torch.bool, device=adj_bool.device)
    queue = [int(src)]
    visited[src] = True
    
    q_ptr = 0
    while q_ptr < len(queue):
        u = queue[q_ptr]
        q_ptr += 1
        nbrs = torch.where(adj_bool[u])[0]
        if nbrs.numel() == 0:
            continue
        
        for v in nbrs.tolist():
            if visited[v]:
                continue
            if v == tgt:
                return True
            visited[v] = True
            queue.append(v)
    return False

class RGCNEncoder(nn.Module):
    def __init__(self, num_nodes, in_dim, hid_dim, num_rel_mp, num_layers=2, dropout=0.2):
        super().__init__()
        self.node_emb = nn.Parameter(torch.randn(num_nodes, in_dim), requires_grad=True)
        self.task_proj = nn.Linear(384, in_dim) 
        
        self.convs = nn.ModuleList()
        self.convs.append(RGCNConv(in_dim, hid_dim, num_rel_mp))
        for _ in range(num_layers - 1):
            self.convs.append(RGCNConv(hid_dim, hid_dim, num_rel_mp))
        self.dropout = nn.Dropout(dropout)

    def forward(self, task_embedding, base_edge_index, base_edge_type):
        task_info = self.task_proj(task_embedding)
        x = self.node_emb + task_info
        for conv in self.convs:
            x = conv(x, base_edge_index, base_edge_type)
            x = F.relu(x)
            x = self.dropout(x)
        return x

class GraphGenerationPolicy(nn.Module):
    def __init__(self, num_nodes, node_in_dim, hid_dim, num_base_rels, num_predict_rels,
                 prune_strategy='threshold', prune_threshold=0.1, prune_top_k=None,
                 min_node_degree=1, keep_start_node=True, edges_per_type=2):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_predict_rels = num_predict_rels
        
        self.prune_strategy = prune_strategy
        self.prune_threshold = prune_threshold
        self.prune_top_k = prune_top_k
        self.min_node_degree = min_node_degree
        self.keep_start_node = keep_start_node
        self.edges_per_type = edges_per_type
        
        self.encoder = RGCNEncoder(num_nodes, node_in_dim, hid_dim, num_base_rels)
        
        self.edge_classifier = nn.Sequential(
            nn.Linear(hid_dim * 2, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, num_predict_rels + 1) 
        )
        
        self.order_scorer = nn.Linear(hid_dim, 1)

    def forward(self, task_embedding, base_edge_index, base_edge_type, temperature=1.0):
        node_embeddings = self.encoder(task_embedding, base_edge_index, base_edge_type)
        
        order_logits = self.order_scorer(node_embeddings).squeeze(-1)
        _, node_order = torch.sort(order_logits, descending=True)
        
        no_edge_idx = self.num_predict_rels 
        dir_adj = torch.zeros(self.num_nodes, self.num_nodes, dtype=torch.bool, device=node_embeddings.device)
        adj_matrices = torch.zeros(self.num_predict_rels, self.num_nodes, self.num_nodes, device=node_embeddings.device)
        
        all_edge_log_probs = []
        all_entropies = []
        
        actual_edge_type_counts = torch.zeros(self.num_predict_rels, device=node_embeddings.device)

        forward_rel_types = {0, 2}
        for i in range(1, self.num_nodes):
            current_node_idx = node_order[i]
            candidate_parent_indices = node_order[:i]
            
            h_current = node_embeddings[current_node_idx]
            h_parents = node_embeddings[candidate_parent_indices]
            
            h_current_expanded = h_current.expand_as(h_parents)
            classifier_input = torch.cat([h_parents, h_current_expanded], dim=1)
            
            all_logits = self.edge_classifier(classifier_input)

            for j, parent_idx in enumerate(candidate_parent_indices):
                parent_idx = int(parent_idx.item())
                logits_j = all_logits[j]

                mask = torch.zeros_like(logits_j, dtype=torch.bool)
                
                if dir_adj[parent_idx, current_node_idx] or dir_adj[current_node_idx, parent_idx]:
                    mask[:no_edge_idx] = True
                else:
                    for r in range(self.num_predict_rels):
                        u, v = -1, -1
                        if r in forward_rel_types:
                            u, v = parent_idx, current_node_idx
                        if u != -1 and _is_reachable(dir_adj, v, u):
                            mask[r] = True

                logits_j.masked_fill_(mask, -float('inf'))

                dist = Categorical(logits=logits_j / temperature)
                sampled_rel = dist.sample()
                
                all_edge_log_probs.append(dist.log_prob(sampled_rel))
                all_entropies.append(dist.entropy())

                if sampled_rel < no_edge_idx:
                    rel_type = int(sampled_rel.item())
                    actual_edge_type_counts[rel_type] += 1
                    
                    edge_prob = torch.exp(dist.log_prob(sampled_rel))
                    
                    adj_matrices[rel_type, parent_idx, current_node_idx] = edge_prob.item()
                    dir_adj[parent_idx, current_node_idx] = True     
        if self.prune_strategy != 'none':
            adj_matrices, pruned_log_probs = self._prune_edges(
                adj_matrices, all_edge_log_probs, node_order, node_embeddings
            )
            total_log_prob = torch.stack(pruned_log_probs).sum() if pruned_log_probs else torch.tensor(0.0, device=node_embeddings.device)
        else:
            total_log_prob = torch.stack(all_edge_log_probs).sum()
        
        avg_entropy = torch.stack(all_entropies).mean() if all_entropies else torch.tensor(0.0)
        actual_edge_type_counts = torch.zeros(self.num_predict_rels, device=node_embeddings.device)
        for r in range(self.num_predict_rels):
            actual_edge_type_counts[r] = (adj_matrices[r] > 0).float().sum()
        
        num_unique_types_used = (actual_edge_type_counts > 0).float().sum()
        coverage_score = num_unique_types_used / self.num_predict_rels
        
        return adj_matrices, total_log_prob, coverage_score, avg_entropy

    def _prune_edges(self, adj_matrices, all_edge_log_probs, node_order, node_embeddings):
        device = adj_matrices.device
        num_rel = adj_matrices.size(0)
        num_nodes = adj_matrices.size(1)
        
        edges_info = []
        log_prob_idx = 0
        
        for i in range(1, num_nodes):
            current_node = node_order[i]
            for j in range(i):
                parent_node = node_order[j]
                for r in range(num_rel):
                    weight = adj_matrices[r, parent_node, current_node].item()
                    if weight > 0:
                        edges_info.append({
                            'rel': r,
                            'src': int(parent_node.item()),
                            'tgt': int(current_node.item()),
                            'weight': weight,
                            'log_prob_idx': log_prob_idx
                        })
                log_prob_idx += 1
        
        if not edges_info:
            return adj_matrices, all_edge_log_probs
        if self.prune_strategy == 'threshold':
            keep_edges = self._prune_by_threshold(edges_info)
        elif self.prune_strategy == 'top_k':
            keep_edges = self._prune_by_top_k(edges_info)
        elif self.prune_strategy == 'balanced_top_k':
            keep_edges = self._prune_by_balanced_top_k(edges_info)
        elif self.prune_strategy == 'adaptive':
            keep_edges = self._prune_adaptive(edges_info, node_order)
        else:
            keep_edges = edges_info

        if self.min_node_degree > 0:
            keep_edges = self._prune_by_degree(keep_edges, node_order)

        new_adj_matrices = torch.zeros_like(adj_matrices)
        kept_log_probs = []
        
        for edge in keep_edges:
            new_adj_matrices[edge['rel'], edge['src'], edge['tgt']] = 1.0
            kept_log_probs.append(all_edge_log_probs[edge['log_prob_idx']])
        
        return new_adj_matrices, kept_log_probs
    
    def _prune_by_threshold(self, edges_info):
        keep_edges = []
        start_node_idx = 0
        
        for edge in edges_info:
            if edge['weight'] >= self.prune_threshold:
                keep_edges.append(edge)
            elif self.keep_start_node and (edge['src'] == start_node_idx or edge['tgt'] == start_node_idx):
                keep_edges.append(edge)
        
        return keep_edges
    
    def _prune_by_top_k(self, edges_info):
        k = self.prune_top_k if self.prune_top_k else len(edges_info)
        k = min(k, len(edges_info))

        sorted_edges = sorted(edges_info, key=lambda x: x['weight'], reverse=True)
        return sorted_edges[:k]
    
    def _prune_by_balanced_top_k(self, edges_info):
        edges_by_type = {}
        for edge in edges_info:
            rel_type = edge['rel']
            if rel_type not in edges_by_type:
                edges_by_type[rel_type] = []
            edges_by_type[rel_type].append(edge)
        
        keep_edges = []

        for rel_type in range(self.num_predict_rels):
            if rel_type in edges_by_type:
                type_edges = sorted(edges_by_type[rel_type], 
                                   key=lambda x: x['weight'], 
                                   reverse=True)
                if rel_type == 1:
                    num_to_select = 1
                elif rel_type == 2:
                    num_to_select = 1
                else:
                    num_to_select = 1
                keep_edges.extend(type_edges[:num_to_select])
        
        return keep_edges
    
    def _prune_adaptive(self, edges_info, node_order):
        keep_edges = []
        start_node = int(node_order[0].item())
        
        high_conf_edges = [e for e in edges_info if e['weight'] >= self.prune_threshold]
        
        adj_bool = torch.zeros(self.num_nodes, self.num_nodes, dtype=torch.bool)
        for edge in high_conf_edges:
            adj_bool[edge['src'], edge['tgt']] = True
        keep_edges = high_conf_edges.copy()
        
        return keep_edges
    
    def _prune_by_degree(self, edges_info, node_order):
        if not edges_info:
            return edges_info
        node_degree = torch.zeros(self.num_nodes, dtype=torch.long)
        for edge in edges_info:
            node_degree[edge['src']] += 1
            node_degree[edge['tgt']] += 1

        keep_edges = []
        start_node = int(node_order[0].item())
        
        for edge in edges_info:
            src_degree = node_degree[edge['src']].item()
            tgt_degree = node_degree[edge['tgt']].item()

            if (src_degree >= self.min_node_degree and tgt_degree >= self.min_node_degree) or \
               (self.keep_start_node and (edge['src'] == start_node or edge['tgt'] == start_node)):
                keep_edges.append(edge)
        
        return keep_edges