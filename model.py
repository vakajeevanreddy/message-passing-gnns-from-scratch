"""
Message-Passing GNNs from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - edges_to_coo
def edges_to_coo(edge_list, num_nodes=None):
    # TODO: Convert a list of (src, dst) edge pairs into COO-format src/dst tensors.
    if isinstance(edge_list,list):
        if len(edge_list) == 0:
            edge_tensor = torch.empty((0,2),dtype = torch.long)
        else:
            edge_tensor = torch.tensor(edge_list, dtype = torch.long)
    elif isinstance(edge_list, torch.Tensor):
        edge_tensor = edge_list
    else:
        raise TypeError("edge_list must be a list of pairs or a LongTensor of shape [E, 2].")
    src = edge_tensor[:,0] if edge_tensor.numel() > 0 else torch.empty(0,dtype = torch.long)
    dst = edge_tensor[:,1] if edge_tensor.numel() > 0 else torch.empty(0,dtype = torch.long)

    if num_nodes is None:
        if edge_tensor.numel() == 0:
            num_nodes = 0
        else:
            num_nodes = int(torch.max(edge_tensor).item())+1
    return src,dst,num_nodes

# Step 2 - add_self_loops
def add_self_loops(src, dst, num_nodes):
    """Append self-loop edges (i, i) for every node to COO edge indices.

    Args:
        src: LongTensor [E] source node indices.
        dst: LongTensor [E] destination node indices.
        num_nodes: int, number of nodes in the graph.

    Returns:
        src_out: LongTensor [E + num_nodes]
        dst_out: LongTensor [E + num_nodes]
    """
    # TODO: Append self-loop edges (i, i) for every node to the COO tensors
    loop_index = torch.arange(num_nodes,dtype = src.dtype , device = src.device)
    src_ext = torch.cat([src,loop_index], dim = 0)
    dst_ext = torch.cat([dst, loop_index] , dim = 0)
    return src_ext,dst_ext

# Step 3 - compute_node_degrees
def compute_node_degrees(src, dst, num_nodes, edge_weight=None):
    """Compute per-node in-degrees (optionally weighted) from COO edges.

    Args:
        src (LongTensor): Source node indices of shape [E].
        dst (LongTensor): Destination node indices of shape [E].
        num_nodes (int): Number of nodes N.
        edge_weight (FloatTensor, optional): Per-edge weights of shape [E].

    Returns:
        FloatTensor: In-degrees of shape [N].
    """
    # TODO: Compute per-node in-degrees by scattering onto destination nodes
    E = dst.size(0)
    if edge_weight is None:
        edge_weight = torch.ones(E,dtype = torch.float,device = dst.device)
    else:
        edge_weight = edge_weight.to(dtype = torch.float, device = dst.device)
    degrees = torch.zeros(num_nodes,dtype = torch.float, device = dst.device)
    degrees.scatter_add_(0,dst,edge_weight)
    return degrees

# Step 4 - symmetric_normalize_edge_weights
def symmetric_normalize_edge_weights(src, dst, num_nodes, edge_weight=None):
    """Compute symmetrically normalized edge weights w_ij / sqrt(d_i * d_j).

    Args:
        src (LongTensor): Source node indices of shape [E].
        dst (LongTensor): Destination node indices of shape [E].
        num_nodes (int): Number of nodes N.
        edge_weight (FloatTensor, optional): Per-edge weights of shape [E].
            Defaults to all ones (float32) when None.

    Returns:
        FloatTensor: Symmetrically normalized weights of shape [E].
    """
    # TODO: Compute symmetrically normalized edge weights for GCN-style propagation.
    E = src.size(0)
    if edge_weight is None:
        edge_weight = torch.ones(E, dtype = torch.float32, device = src.device)
    else:
        edge_weight = edge_weight.to(dtype = torch.float32 , device = src.device)
    degrees = torch.zeros(num_nodes,dtype = torch.float32 , device = src.device)
    degrees.scatter_add_(0,dst,edge_weight)

    inv_sqrt_deg = torch.zeros_like(degrees)
    nonzero_mask = degrees > 0
    inv_sqrt_deg[nonzero_mask] = degrees[nonzero_mask].pow(-0.5)

    norm_weights = edge_weight * inv_sqrt_deg[src] * inv_sqrt_deg[dst]
    return norm_weights

# Step 5 - gather_source_node_features
def gather_source_node_features(node_features, src):
    # TODO: Return edge-aligned source feature rows (E, F) from node_features.
    edge_src_features = node_features[src]
    return edge_src_features

# Step 6 - scatter_sum_to_nodes
def scatter_sum_to_nodes(edge_features, dst, num_nodes):
    """Scatter-sum edge features onto destination nodes to produce per-node aggregated vectors.

    Args:
        edge_features: FloatTensor of shape (E, F) with one feature row per edge.
        dst: LongTensor of shape (E,) with destination node index for each edge.
        num_nodes: int, number of nodes N in the graph.

    Returns:
        FloatTensor of shape (N, F); row j is the sum of edge features with dst == j.
    """
    # TODO: Scatter-sum edge features onto destination nodes to produce per-node vectors
    N,F = num_nodes,edge_features.size(1)
    node_features = torch.zeros((N,F), dtype = edge_features.dtype, device = edge_features.device)
    node_features.index_add_(0,dst,edge_features)
    return node_features

# Step 7 - scatter_mean_to_nodes
def scatter_mean_to_nodes(edge_features, dst, num_nodes):
    # TODO: Scatter-mean edge features onto destination nodes (sum then divide by in-degree).
    N,F = num_nodes,edge_features.size(1)
    node_sum = torch.zeros((N,F),dtype = edge_features.dtype , device = edge_features.device)
    node_sum.index_add_(0,dst,edge_features)

    counts = torch.zeros(N,dtype = edge_features.dtype, device = edge_features.device)
    ones = torch.ones(dst.size(0), dtype = edge_features.dtype, device = edge_features.device)
    counts.index_add_(0,dst,ones)

    node_mean = torch.zeros_like(node_sum)
    nonzero_mask = counts > 0
    node_mean[nonzero_mask] = node_sum[nonzero_mask]/counts[nonzero_mask].unsqueeze(-1)
    return node_mean

# Step 8 - scatter_max_to_nodes
def scatter_max_to_nodes(edge_features, dst, num_nodes):
    # TODO: Scatter-max edge features onto destination nodes (elementwise max).
    N,F = num_nodes,edge_features.size(1)
    node_features = torch.full((N,F), float('-inf'),dtype = edge_features.dtype , device = edge_features.device)
    for i in range(edge_features.size(0)):
        nodes = dst[i]
        node_features[nodes] = torch.maximum(node_features[nodes], edge_features[i])
    return node_features

# Step 9 - compute_messages
def compute_messages(node_features, src, dst, message_fn, edge_attr=None):
    """Build per-edge messages via gather + message_fn.

    Args:
        node_features: FloatTensor of shape (N, F).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        message_fn: callable(src_feats, dst_feats[, edge_attr]) -> messages.
        edge_attr: optional FloatTensor of shape (E, Fe).

    Returns:
        messages: FloatTensor of shape (E, M).
    """
    # TODO: Build per-edge messages by gathering features and applying message_fn
    src_features = node_features[src]

    Dst_features = node_features[dst]
    if edge_attr is not None:
        messages = message_fn(src_features,Dst_features,edge_attr)
    else:
        messages = message_fn(src_features,Dst_features)
    return messages

# Step 10 - aggregate_messages
def aggregate_messages(messages, dst, num_nodes, aggr='sum'):
    """Aggregate edge messages onto destination nodes using sum, mean, or max.

    Args:
        messages: FloatTensor of shape (E, M) with one message vector per edge.
        dst: LongTensor of shape (E,) with destination node index for each edge.
        num_nodes: int, number of nodes N in the graph.
        aggr: str in {'sum', 'mean', 'max'} selecting the reduction.

    Returns:
        FloatTensor of shape (N, M); row j is the aggregated message for node j.
    """
    # TODO: Aggregate edge messages onto destination nodes via sum/mean/max...
    E,M = messages.shape
    out = torch.zeros(num_nodes,M,device = messages.device)

    if aggr == 'sum':
        out.index_add_(0,dst,messages)
    elif aggr == 'mean':
        out.index_add_(0,dst,messages)
        counts = torch.bincount(dst,minlength = num_nodes).clamp(min=1).unsqueeze(-1)
        out = out/counts
    elif aggr == 'max':
        out.fill_(float('-inf'))
        if hasattr(torch.Tensor, "scatter_reduce_"):
            out.scatter_reduce_(0,dst.unsqueeze(-1).expand(-1,M),messages,reduce = "amax", include_self = True)
        else:
            out.index_put_((dst,), messages, accumulate = True)
        out[out == float('-inf')] = 0.0
    else:
        raise ValueError(f"Unsupported aggregation mode: {aggr}")
    return out

# Step 11 - update_node_features
def update_node_features(node_features, aggregated, update_fn):
    # TODO: Implement update_node_features to fuse each node's current state with its aggregated...
    if not callable(update_fn):
        raise ValueError("update_fn must be a callable function")
    return update_fn(node_features,aggregated)

# Step 12 - message_passing_layer
import torch

def message_passing_layer(node_features, src, dst,
                          message_fn, update_fn,
                          aggr='sum', edge_attr=None):
    """Run one full Gilmer MPNN step: message, aggregate, and update."""
    # Step 1: compute messages
    src_feats = node_features[src]
    dst_feats = node_features[dst]
    if edge_attr is not None:
        messages = message_fn(src_feats, dst_feats, edge_attr)
    else:
        messages = message_fn(src_feats, dst_feats)

    # Step 2: aggregate messages
    N = node_features.size(0)
    M = messages.size(1)
    aggregated = torch.zeros(N, M, device=node_features.device)

    if aggr == 'sum':
        aggregated.index_add_(0, dst, messages)

    elif aggr == 'mean':
        aggregated.index_add_(0, dst, messages)
        counts = torch.bincount(dst, minlength=N).clamp(min=1).unsqueeze(-1)
        aggregated = aggregated / counts

    elif aggr == 'max':
        aggregated.fill_(float('-inf'))
        if hasattr(torch.Tensor, "scatter_reduce_"):
            aggregated.scatter_reduce_(0,dst.unsqueeze(-1).expand(-1, M),
                messages,reduce="amax",include_self=True)
        else:
            aggregated.index_put_((dst,), messages, accumulate=True)
        # Replace -inf with 0 for nodes with no incoming messages
        #aggregated[aggregated == float('-inf')] = 0.0
    else:
        raise ValueError(f"Unsupported aggregation mode: {aggr}")
    # Step 3: update node features
    updated = update_fn(node_features, aggregated)
    return updated

# Step 13 - stack_message_passing_layers
def stack_message_passing_layers(node_features, src, dst, layers, edge_attr=None):
    """Apply a sequence of message-passing layer callables to produce deep node embeddings.

    Args:
        node_features: FloatTensor of shape (N, F).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        layers: list of callables, each
            layer(node_features, src, dst, edge_attr=None) -> Tensor (N, H_i).
        edge_attr: optional FloatTensor of shape (E, Fe).

    Returns:
        embeddings: FloatTensor of shape (N, H), final layer output.
        all_layer_outputs: list of FloatTensors, one per layer (N, H_i).
    """
    # TODO: Apply a sequence of MP layer callables; return final + intermediates
    if not layers:
        return node_features,[]

    intermidates = []
    h = node_features
    for layer in layers:
        h = layer(h,src,dst,edge_attr)
        intermidates.append(h)
    return h,intermidates

# Step 14 - gcn_renormalize_adjacency
import torch

def gcn_renormalize_adjacency(src, dst, num_nodes):
    device = src.device

    # Step 1: add self-loops
    self_loops = torch.arange(num_nodes, device=device)
    src_hat = torch.cat([src, self_loops])
    dst_hat = torch.cat([dst, self_loops])

    # Step 2: compute degree (incoming edges + self-loops)
    deg = torch.zeros(num_nodes, device=device)
    deg.scatter_add_(0, dst_hat, torch.ones_like(dst_hat, dtype=deg.dtype))

    # Step 3: symmetric normalization
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    norm_weight = deg_inv_sqrt[src_hat] * deg_inv_sqrt[dst_hat]

    return src_hat, dst_hat, norm_weight

# Step 15 - gcn_linear_transform
def gcn_linear_transform(node_features, weight, bias=None):
    """Apply the GCN linear feature transform X @ W (+ bias).

    Args:
        node_features: FloatTensor of shape (N, Fin).
        weight: FloatTensor of shape (Fin, Fout).
        bias: optional FloatTensor of shape (Fout).

    Returns:
        FloatTensor of shape (N, Fout).
    """
    # TODO: compute the matrix product and optionally add a bias vector
    transformed = node_features @ weight
    if bias is not None:
        transformed = transformed + bias
    return transformed

# Step 16 - gcn_layer_forward
def gcn_layer_forward(node_features, src, dst, weight, bias=None, num_nodes=None, activation=None):
    """Forward pass of one GCN layer: renormalize, transform, propagate.

    Args:
        node_features: FloatTensor of shape (N, Fin).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        weight: FloatTensor of shape (Fin, Fout).
        bias: optional FloatTensor of shape (Fout,).
        num_nodes: optional int N; defaults to node_features.shape[0].
        activation: optional callable applied to the output.

    Returns:
        FloatTensor of shape (N, Fout).
    """
    # TODO: Forward pass of one GCN layer: renormalize, transform, propagate...
    if num_nodes is None:
        num_nodes = node_features.size(0)
    src_hat,dst_hat,norm_weight = gcn_renormalize_adjacency(src,dst,num_nodes)
    h = gcn_linear_transform(node_features,weight)
    h_src = h[src_hat]
    messages = norm_weight.unsqueeze(-1) * h_src
    out = torch.zeros_like(h)
    out.scatter_add_(0,dst_hat.unsqueeze(-1).expand(-1,h.size(1)), messages)
    if bias is not None:
        out = out + bias
    if activation is not None:
        out = activation(out)
    return out

# Step 17 - init_gcn_parameters
import math
import torch
def init_gcn_parameters(in_dim, out_dim, with_bias=True, seed=None):
    # TODO: Initialize GCN weight (and optional bias) with Glorot-style uniform...
    if seed is not None:
        torch.manual_seed(seed)
    a = math.sqrt(6.0/(in_dim+out_dim))
    weight = torch.empty(in_dim,out_dim).uniform_(-a,a)
    params = {'weight': weight}
    if with_bias:
        bias = torch.zeros(out_dim)
        params['bias'] = bias
    return params

# Step 18 - gcn_stack_forward (not yet solved)
# TODO: implement

# Step 19 - gat_attention_logits
import torch.nn.functional as F

def gat_attention_logits(node_features, src, dst, attn_src, attn_dst, weight):
    """Compute unnormalized GAT attention logits and transformed features.

    Args:
        node_features: FloatTensor of shape (N, Fin).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        attn_src: FloatTensor of shape (Fout,) source attention vector.
        attn_dst: FloatTensor of shape (Fout,) destination attention vector.
        weight: FloatTensor of shape (Fin, Fout) shared linear transform.

    Returns:
        logits: FloatTensor of shape (E,) unnormalized attention scores.
        transformed: FloatTensor of shape (N, Fout) linearly transformed nodes.
    """
    # TODO: return per-edge LeakyReLU attention logits and transformed features
    h_prime = node_features @ weight
    h_src = h_prime[src]
    h_dst = h_prime[dst]
    alpha_src = (h_src * attn_src).sum(dim = -1)
    alpha_dst = (h_dst * attn_dst).sum(dim = -1)
    logits = F.leaky_relu(alpha_src + alpha_dst, negative_slope = 0.2)
    return logits,h_prime

# Step 20 - gat_masked_neighbor_softmax
def gat_masked_neighbor_softmax(logits, dst, num_nodes):
    """Numerically stable softmax of attention logits over each dest node's neighbors.

    Args:
        logits: FloatTensor of shape (E,) with one unnormalized attention logit per edge.
        dst: LongTensor of shape (E,) with destination node index for each edge.
        num_nodes: int, number of nodes N in the graph.

    Returns:
        FloatTensor of shape (E,) with attention coefficients that sum to 1 over
        each destination's incoming edges.
    """
    # TODO: Numerically stable softmax of attention logits over each dest node's neighbors
    max_per_dst = torch.full((num_nodes,),float('-inf'))
    max_per_dst.scatter_reduce_(0,dst,logits,reduce="amax",include_self = True)
    stable_logits = logits - max_per_dst[dst]
    exp_logits = stable_logits.exp()
    sum_per_dst = torch.zeros(num_nodes,device = logits.device)
    sum_per_dst.scatter_add_(0,dst,exp_logits)
    attn_coeffs = exp_logits / sum_per_dst[dst]
    return attn_coeffs

# Step 21 - gat_head_forward
def gat_head_forward(node_features, src, dst, weight, attn_src, attn_dst, bias=None, num_nodes=None, activation=None):
    """Forward pass of a single GAT attention head.

    Args:
        node_features: FloatTensor of shape (N, Fin).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        weight: FloatTensor of shape (Fin, Fout) shared linear transform.
        attn_src: FloatTensor of shape (Fout,) source attention vector.
        attn_dst: FloatTensor of shape (Fout,) destination attention vector.
        bias: optional FloatTensor of shape (Fout,).
        num_nodes: optional int N; inferred from node_features if None.
        activation: optional callable applied to the head output.

    Returns:
        head_out: FloatTensor of shape (N, Fout).
        attn_coeffs: FloatTensor of shape (E,) attention coefficients.
    """
    # TODO: Forward pass of a single GAT attention head: transform, coeffs, aggregate...
    if num_nodes is None:
        num_nodes = node_features.size(0)
    h = node_features @ weight
    h_src = h[src]
    h_dst = h[dst]
    logits = (h_src * attn_src).sum(dim = -1) + (h_dst * attn_dst).sum(dim=-1)
    logits = F.leaky_relu(logits, negative_slope=0.2)


    max_per_dst = torch.full((num_nodes,),float('-inf'))
    max_per_dst.scatter_reduce_(0,dst,logits,reduce = "amax",include_self = True)
    stable_logits = logits - max_per_dst[dst]
    exp_logits = stable_logits.exp()
    sum_per_dst = torch.zeros(num_nodes,device = logits.device)
    sum_per_dst.scatter_add_(0,dst,exp_logits)
    alpha = exp_logits / sum_per_dst[dst]

    out = torch.zeros((num_nodes,weight.size(1)),device = h.device)
    out.scatter_add_(0,dst.unsqueeze(-1).expand(-1,h.size(1)),alpha.unsqueeze(-1)*h_src)
    if bias is not None:
        out = out + bias
    if activation is not None:
        out = activation(out)
    return out,alpha

# Step 22 - merge_gat_heads (not yet solved)
# TODO: implement

# Step 23 - gat_layer_forward (not yet solved)
# TODO: implement

# Step 24 - init_gat_parameters (not yet solved)
# TODO: implement

# Step 25 - gat_stack_forward (not yet solved)
# TODO: implement

# Step 26 - global_mean_pool (not yet solved)
# TODO: implement

# Step 27 - global_sum_pool (not yet solved)
# TODO: implement

# Step 28 - global_max_pool (not yet solved)
# TODO: implement

# Step 29 - global_mean_max_pool (not yet solved)
# TODO: implement

# Step 30 - node_classification_head (not yet solved)
# TODO: implement

# Step 31 - graph_regression_head (not yet solved)
# TODO: implement

# Step 32 - generate_sbm_graph (not yet solved)
# TODO: implement

# Step 33 - build_node_classification_dataset (not yet solved)
# TODO: implement

# Step 34 - generate_molecule_like_graph (not yet solved)
# TODO: implement

# Step 35 - build_graph_regression_dataset (not yet solved)
# TODO: implement

# Step 36 - collate_graph_batch (not yet solved)
# TODO: implement

# Step 37 - cross_entropy_loss (not yet solved)
# TODO: implement

# Step 38 - mse_loss (not yet solved)
# TODO: implement

# Step 39 - accuracy_metric (not yet solved)
# TODO: implement

# Step 40 - mae_metric (not yet solved)
# TODO: implement

# Step 41 - gnn_train_step (not yet solved)
# TODO: implement

# Step 42 - train_node_classifier (not yet solved)
# TODO: implement

# Step 43 - train_graph_regressor (not yet solved)
# TODO: implement

# Step 44 - representation_similarity (not yet solved)
# TODO: implement

# Step 45 - oversmoothing_diagnostic (not yet solved)
# TODO: implement

# Step 46 - mpnn_gnn_experiment (not yet solved)
# TODO: implement

