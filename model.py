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

# Step 18 - gcn_stack_forward
def gcn_stack_forward(node_features, src, dst, param_list, activations=None, num_nodes=None):
    """Run a stack of GCN layers to produce deep node embeddings.

    Args:
        node_features: FloatTensor of shape (N, F0).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        param_list: list of dicts, each with 'weight' (Fin, Fout) and optional 'bias' (Fout,).
        activations: optional list of callables or None, one per layer.
        num_nodes: optional int N; defaults to node_features.shape[0].

    Returns:
        embeddings: FloatTensor of shape (N, FL), the final layer output.
        all_layer_outputs: list of FloatTensor outputs after each layer.
    """
    # TODO: Run a stack of GCN layers to produce deep node embeddings
    if num_nodes is None:
        num_nodes = node_features.size(0)
    if activations is None:
        activations = [None] * len(param_list)
    h = node_features
    all_layer_outputs = []
    for layer_index,params in enumerate(param_list):
        weight = params['weight']
        bias = params.get('bias', None)
        activation = activations[layer_index]
        h = gcn_layer_forward(h,src,dst,weight,bias = bias,
                            num_nodes = num_nodes, activation = activation)
        all_layer_outputs.append(h)
    embeddings = h
    return embeddings,all_layer_outputs

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

# Step 22 - merge_gat_heads
def merge_gat_heads(head_outputs, mode='concat'):
    # TODO: Merge multi-head GAT outputs into one node-feature tensor.
    if isinstance(head_outputs,(list,tuple)):
        head_outputs = torch.stack(head_outputs)
    if head_outputs.dim() != 3:
        raise ValueError("Expected input of shape[H,N,F] or list [N,F] tensors")
    H,N,F = head_outputs.shape
    if mode == "concat":
        return head_outputs.permute(1,0,2).reshape(N,H*F)

    if mode == "mean":
        return head_outputs.mean(dim = 0)
    else:
        raise ValueError(f"Unsupported mode '{mode}' use 'concat' or 'mean' ")

# Step 23 - gat_layer_forward
def gat_layer_forward(node_features, src, dst, head_params, merge_mode='concat', num_nodes=None, activation=None):
    """Multi-head GAT layer: run each head, merge, optional activation.

    Args:
        node_features: FloatTensor (N, Fin).
        src: LongTensor (E,) source indices.
        dst: LongTensor (E,) destination indices.
        head_params: list of dicts with keys weight, attn_src, attn_dst,
            and optional bias for each head.
        merge_mode: 'concat' or 'mean'.
        num_nodes: optional int N; inferred from node_features if None.
        activation: optional callable applied after merging heads.

    Returns:
        out: FloatTensor (N, F_merged).
        all_attn: list of FloatTensor (E,) attention coeffs per head.
    """
    # TODO: run each head, merge outputs, apply optional nonlinearity...
    if num_nodes is None:
        num_nodes = node_features.shape[0]
    head_outputs = []
    all_attn = []
    for params in head_params:
        head_out,attn_coeffs = gat_head_forward(
            node_features,src,dst,params["weight"],params["attn_src"],params["attn_dst"],
                                    bias = params.get("bias", None),
                                    num_nodes = num_nodes,activation = None)
        head_outputs.append(head_out)
        all_attn.append(attn_coeffs)
    out = merge_gat_heads(head_outputs,mode = merge_mode)
    if activation is not None:
        out = activation(out)
    return out,all_attn

# Step 24 - init_gat_parameters
import torch
import math

def init_gat_parameters(in_dim, out_dim, num_heads=1, with_bias=True, seed=None):
    # Initialize multi-head GAT parameters with Glorot-style initialization.
    if seed is not None:
        torch.manual_seed(seed)

    params = []
    a_w = math.sqrt(6.0 / (in_dim + out_dim))
    a_attn = math.sqrt(6.0 / (out_dim + 1))

    for _ in range(num_heads):
        weight = torch.empty((in_dim, out_dim), dtype=torch.float32).uniform_(-a_w, a_w)
        weight.requires_grad_(True)

        attn_src = torch.empty((out_dim,), dtype=torch.float32).uniform_(-a_attn, a_attn)
        attn_src.requires_grad_(True)

        attn_dst = torch.empty((out_dim,), dtype=torch.float32).uniform_(-a_attn, a_attn)
        attn_dst.requires_grad_(True)

        head_dict = {
            "weight": weight,
            "attn_src": attn_src,
            "attn_dst": attn_dst
        }

        if with_bias:
            bias = torch.zeros((out_dim,), dtype=torch.float32)
            bias.requires_grad_(True)
            head_dict["bias"] = bias

        params.append(head_dict)

    return params

# Step 25 - gat_stack_forward
def gat_stack_forward(node_features, src, dst, layer_param_list, merge_modes=None, activations=None, num_nodes=None):
    """Run a stack of multi-head GAT layers.

    Args:
        node_features: FloatTensor (N, F0).
        src: LongTensor (E,) source indices.
        dst: LongTensor (E,) destination indices.
        layer_param_list: list of length L; each entry is a head_params list
            for gat_layer_forward.
        merge_modes: optional list of L merge mode strings ('concat' or 'mean').
            Defaults to 'concat' for every layer.
        activations: optional list of L callables or None. Defaults to no
            activation for every layer.
        num_nodes: optional int N; inferred from node_features if None.

    Returns:
        embeddings: FloatTensor (N, FL) final layer output.
        all_layer_outputs: list of L FloatTensors, the output after each layer.
    """
    # TODO: Run a stack of multi-head GAT layers for deep node embeddings.
    num_layers = len(layer_param_list)

    if merge_modes is None:
        merge_modes = ["concat"]*num_layers
    if activations is None:
        activations = [None] * num_layers
    if num_nodes is None:
        num_nodes = node_features.shape[0]
    all_layer_outputs = []
    out = node_features
    for i,head_params in enumerate(layer_param_list):
        out,_ = gat_layer_forward(out,src,dst,head_params,
                num_nodes = out.shape[0],
                merge_mode = merge_modes[i],
                activation = activations[i])
        all_layer_outputs.append(out)
    return out,all_layer_outputs

# Step 26 - global_mean_pool
def global_mean_pool(node_features, batch_index, num_graphs=None):
    """Globally mean-pool node features into one graph-level vector per graph.

    Args:
        node_features: FloatTensor of shape (N, F) with one feature row per node.
        batch_index: LongTensor of shape (N,) mapping each node to a graph id in
            {0, ..., B-1}.
        num_graphs: Optional int B. If None, inferred as batch_index.max() + 1.

    Returns:
        FloatTensor of shape (B, F); row b is the mean of node features with
        batch_index == b.
    """
    # TODO: Mean-pool node features into one graph-level vector per graph...
    if num_graphs is None:
        num_graphs = int(batch_index.max().item()) + 1
    summed = scatter_sum_to_nodes(node_features,batch_index,num_graphs)
    ones = torch.ones(batch_index.size(0), 1 , device = node_features.device)
    counts = scatter_sum_to_nodes(ones,batch_index,num_graphs)

    counts = counts.clamp(min = 1)
    mean_pooled = summed / counts
    return mean_pooled

# Step 27 - global_sum_pool
def global_sum_pool(node_features, batch_index, num_graphs=None):
    """Globally sum-pool node features into one graph-level vector per graph.

    Args:
        node_features: FloatTensor of shape (N, F) with one row per node.
        batch_index: LongTensor of shape (N,) mapping each node to a graph id
            in 0 .. B-1.
        num_graphs: optional int B. If None, inferred as max(batch_index) + 1.

    Returns:
        FloatTensor of shape (B, F); row g is the sum of node features with
        batch_index == g.
    """
    # TODO: sum-pool node features into one graph-level vector per graph
    if num_graphs is None:
        num_graphs = int(batch_index.max().item()) + 1
    N,F = node_features.shape
    pooled = torch.zeros(num_graphs,F,dtype = node_features.dtype, device = node_features.device)
    pooled.index_add_(0,batch_index,node_features)
    return pooled

# Step 28 - global_max_pool
def global_max_pool(node_features, batch_index, num_graphs=None):
    # TODO: Globally max-pool node features into one graph-level vector per graph.
    if num_graphs is None:
        num_graphs = int(batch_index.max().item()) + 1
    N,F = node_features.shape
    pooled = torch.full((num_graphs,F),float('-inf'),
                dtype = node_features.dtype,
                device = node_features.device)
    pooled = pooled.scatter_reduce(
    0,
    batch_index.unsqueeze(-1).expand(-1, F),
    node_features,reduce="amax",include_self=True)
    return pooled

# Step 29 - global_mean_max_pool
def global_mean_max_pool(node_features, batch_index, num_graphs=None):
    """Concatenate global mean and max pooled features into a 2F-dim graph vector.

    Args:
        node_features: FloatTensor of shape (N, F).
        batch_index: LongTensor of shape (N,) with graph ids in {0, ..., B-1}.
        num_graphs: Optional int B. If None, inferred as batch_index.max() + 1.

    Returns:
        FloatTensor of shape (B, 2F); each row is [mean_pool || max_pool].
    """
    # TODO: Concatenate global mean and max pooled features into a 2F-dim vector...
    mean_pooled = global_mean_pool(node_features,batch_index,num_graphs)
    max_pooled = global_max_pool(node_features,batch_index,num_graphs)
    return torch.cat([mean_pooled,max_pooled],dim = -1)

# Step 30 - node_classification_head
def node_classification_head(node_embeddings, weight, bias=None):
    # TODO: Map node embeddings to per-node class logits via a linear head...
    logits = node_embeddings @ weight
    if bias is not None:
        logits = logits + bias
    return logits

# Step 31 - graph_regression_head
def graph_regression_head(graph_embeddings, weight, bias=None):
    # TODO: Map pooled graph embeddings to regression predictions via a linear head.
    preds = graph_embeddings @ weight.T
    if bias is not None:
        preds = preds + bias
    return preds

# Step 32 - generate_sbm_graph
def generate_sbm_graph(num_nodes, num_classes, p_in, p_out, feature_dim, seed=None):
    # TODO: Sample one SBM graph with community labels and random node features.
    if seed is not None:
        torch.manual_seed(seed)
    labels = torch.empty(num_nodes,dtype = torch.long)
    for c in range(num_classes):
        start = c * num_nodes // num_classes
        end = (c + 1) * num_nodes // num_classes
        labels[start:end] = c 
    edges = []
    for i in range(num_nodes):
        for j in range(i+1,num_nodes):
            if labels[i] == labels[j]:
                prob = p_in
            else:
                prob = p_out
            if torch.rand(1).item()<prob:
                edges.append([i,j])
                edges.append([j,i])
    if edges:
        edge_index = torch.tensor(edges, dtype = torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2,0),dtype = torch.long)
    node_features = torch.randn(num_nodes,feature_dim)
    return {
        "node_features" : node_features,
        "edge_index" : edge_index,
        "node_labels" : labels,
        "num_nodes": num_nodes
    }

# Step 33 - build_node_classification_dataset
def build_node_classification_dataset(num_graphs, num_nodes, num_classes, p_in, p_out, feature_dim, seed=None):
    # TODO: Build a list of SBM graphs with consistent schema for node classification.


    graphs = []
    for i in range(num_graphs):
        graph_seed = seed + i if seed is not None else None
        g = generate_sbm_graph(num_nodes = num_nodes,
                            num_classes = num_classes,
                            feature_dim = feature_dim,
                            p_in =p_in,
                            p_out=p_out,
                            seed = graph_seed)
        graphs.append(g)
    return graphs

# Step 34 - generate_molecule_like_graph
def generate_molecule_like_graph(num_nodes, num_node_features, edge_prob=0.3, seed=0):
    # TODO: Synthesize one molecule-like graph with features, edges, and target...
    if seed is not None:
        torch.manual_seed(seed)
    x = torch.randn(num_nodes,num_node_features)
    edges = []
    for i in range(num_nodes):
        for j in range(i+1,num_nodes):
            if torch.rand(1).item() < edge_prob:
                edges.append([i,j])
                edges.append([j,i])
    if edges:
        edge_index = torch.tensor(edges,dtype = torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2,0),dtype = torch.long)
    deg = torch.zeros(num_nodes,dtype = torch.float)
    for i,j in edges:
        deg[i] += 1
    mean_node = x.mean(dim=1)
    y_val = (deg * mean_node).mean()
    y = torch.tensor(y_val,dtype = torch.float)
    return {"x" : x, "edge_index" : edge_index, "y":y}

# Step 35 - build_graph_regression_dataset
def build_graph_regression_dataset(num_graphs, num_nodes_range, num_node_features, edge_prob=0.3, seed=0):
    # TODO: Build a list of molecule-like graphs for graph-level regression.
    lo,hi = num_nodes_range
    graphs = []
    for i in range(num_graphs):
        num_nodes = lo + (i % (hi - lo + 1))
        graph_seed = seed + i
        g = generate_molecule_like_graph(num_nodes = num_nodes,
            num_node_features = num_node_features,
            edge_prob = edge_prob,
            seed = graph_seed)
        graphs.append(g)
    return graphs

# Step 36 - collate_graph_batch
def collate_graph_batch(graphs):
    xs = []
    edge_indexes = []
    ys = []
    batch = []
    node_offset = 0
    for gid, g in enumerate(graphs):
        x = g["x"]
        edge_index = g["edge_index"]
        y = g["y"]
        N_i = x.size(0)

        xs.append(x)
        edge_indexes.append(edge_index + node_offset)
        batch.append(torch.full((N_i,), gid, dtype=torch.long))
        ys.append(torch.as_tensor(y, dtype=torch.float).view(()))

        node_offset += N_i

    x_cat = torch.cat(xs, dim=0)
    edge_index_cat = torch.cat(edge_indexes, dim=1)
    batch_cat = torch.cat(batch, dim=0)
    y_stack = torch.stack(ys, dim=0)

    return {
        "x": x_cat,
        "edge_index": edge_index_cat, 
        "batch": batch_cat,
        "y": y_stack,
    }

# Step 37 - cross_entropy_loss
def cross_entropy_loss(logits, targets):
    # TODO: Compute mean multi-class cross-entropy between logits and targets.
    log_probs = torch.nn.functional.log_softmax(logits,dim = 1)
    loses = -log_probs[torch.arange(logits.size(0)),targets]
    return loses.mean()

# Step 38 - mse_loss
def mse_loss(predictions, targets):
    # TODO: Compute mean squared error between predictions and targets
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    squared_diff = (predictions - targets) ** 2
    return squared_diff.mean()

# Step 39 - accuracy_metric
import numpy as np
def accuracy_metric(logits, targets):
    # TODO: Return the fraction of argmax(logits) predictions matching targets.
    predictions = np.argmax(logits,axis = 1 )
    correct = (predictions == targets).sum()
    accuracy = correct / len(targets)
    return float(accuracy)

# Step 40 - mae_metric
def mae_metric(predictions, targets):
    # TODO: Compute mean absolute error between predicted and target continuous values.
    preds = predictions.reshape(-1)
    targs = targets.reshape(-1)
    abs_diff = torch.abs(preds - targs)
    mae = torch.mean(abs_diff)
    return mae.item()

# Step 41 - gnn_train_step
def gnn_train_step(params, batch, forward_fn, loss_fn, lr):
    # TODO: Run one SGD training step and update params in-place...
    predictions = forward_fn(params,batch)
    targets = batch["y"]
    loss = loss_fn(predictions,targets)
    loss.backward()
    with torch.no_grad():
        for name,p in params.items():
            if p.grad is not None:
                p -= lr * p.grad
                p.grad.zero_()
    return {"loss" : loss.item(),"params":params}

# Step 42 - train_node_classifier
def train_node_classifier(params, dataset, forward_fn, num_epochs, lr, mask_key='train_mask'):
    # TODO: Train a functional node-classification GNN for several epochs on a masked graph
    history = []
    y_all = dataset["y"]
    mask = dataset[mask_key].bool()
    for epoch in range(num_epochs):
        x = dataset["x"]
        edge_index = dataset["edge_index"]
        preds = forward_fn(params,x,edge_index)
        preds_masked = preds[mask]
        targets_masked = y_all[mask]

        loss = F.cross_entropy(preds_masked,targets_masked)
        loss.backward()

        with torch.no_grad():
            for p in params.values():
                if p.grad is not  None:
                    p -= lr * p.grad
                    p.grad.zero_()
        with torch.no_grad():
            logits = forward_fn(params,x,edge_index)
            logits_masked = logits[mask]
            labels_masked = y_all[mask]
            correct = (logits_masked.argmax(dim = - 1) == labels_masked).sum().item()
            acc = correct / mask.sum().item()
        history.append({"loss" : loss.item(),"accuracy" : acc})
    return {"history" : history , "params" : params}

# Step 43 - train_graph_regressor
import torch
import torch.nn.functional as F

def train_graph_regressor(params, graphs, forward_fn, collate_fn=None, num_epochs=20, lr=0.01, batch_size=8):
    """Train a graph regressor over multiple epochs of mini-batches.

    Args:
        params: dict of trainable torch tensors.
        graphs: list of graph dicts with keys x, edge_index, y.
        forward_fn: callable(params, batch) -> predictions.
        collate_fn: callable(list_of_graphs) -> batched graph dict (optional).
        num_epochs: number of training epochs (default 20).
        lr: learning rate for SGD updates (default 0.01).
        batch_size: graphs per mini-batch (default 8).

    Returns:
        history: dict with 'loss' and 'mae' lists of per-epoch floats.
        params: updated parameter dict.
    """
    if collate_fn is None:
        def collate_fn(batch_graphs):
            xs = [torch.as_tensor(g["x"], dtype=torch.float) for g in batch_graphs]
            processed_xs = []
            for x in xs:
                if x.ndim == 1:
                    x = x.unsqueeze(1)
                elif x.ndim > 2:
                    x = x.reshape(x.shape[0], -1)
                processed_xs.append(x)
            
            # Align feature dimensions across graphs in the batch to prevent shape mismatches
            max_feat = max(x.shape[1] for x in processed_xs)
            aligned_xs = []
            for x in processed_xs:
                if x.shape[1] < max_feat:
                    x = F.pad(x, (0, max_feat - x.shape[1]))
                elif x.shape[1] > max_feat:
                    x = x[:, :max_feat]
                aligned_xs.append(x)
            
            x_tensor = torch.cat(aligned_xs, dim=0)

            return {
                "x": x_tensor,
                "edge_index": [g["edge_index"] for g in batch_graphs],
                "y": torch.stack([torch.as_tensor(g["y"], dtype=torch.float) for g in batch_graphs]),
                "batch": torch.cat([torch.full((x.shape[0],), i, dtype=torch.long) for i, x in enumerate(aligned_xs)])
            }

    history = {"loss": [], "mae": []}
    n = len(graphs)
    y_all = torch.stack([torch.as_tensor(g["y"], dtype=torch.float).view(()) for g in graphs])

    for epoch in range(num_epochs):
        total_loss = 0.0
        n_batches = 0
        perm = torch.randperm(n)

        for start in range(0, n, batch_size):
            batch_indices = perm[start:start + batch_size]
            batch_graphs = [graphs[j.item()] for j in batch_indices]
            batch = collate_fn(batch_graphs)
            preds = forward_fn(params, batch).view(-1)
            targets = batch["y"].view(-1)

            loss = F.mse_loss(preds, targets)
            loss.backward()

            with torch.no_grad():
                for p in params.values():
                    if p.grad is not None:
                        p -= lr * p.grad
                        p.grad.zero_()
            total_loss += loss.item()
            n_batches += 1

        mean_loss = total_loss / max(n_batches, 1)
        history["loss"].append(mean_loss)

        with torch.no_grad():
            preds_all = []
            for start in range(0, n, batch_size):
                batch_graphs = graphs[start:start + batch_size]
                batch = collate_fn(batch_graphs)
                preds = forward_fn(params, batch).view(-1)
                preds_all.append(preds)
            preds_all = torch.cat(preds_all)
            mae = (preds_all - y_all).abs().mean().item()

        history["mae"].append(mae)

    return history, params

# Step 44 - representation_similarity (not yet solved)
# TODO: implement

# Step 45 - oversmoothing_diagnostic (not yet solved)
# TODO: implement

# Step 46 - mpnn_gnn_experiment (not yet solved)
# TODO: implement

