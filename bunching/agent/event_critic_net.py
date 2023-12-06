import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, GCNConv
from torch_scatter import scatter_sum
import copy


class Event_Critic_Net(torch.nn.Module):
    def __init__(self, state_size, hidden_size, init_type='default'):
        super().__init__()
        # add_self_loops=False because we add the self node manually with edge_index being [0, 0]
        self.up_conv1 = GATConv(state_size, hidden_size,
                                add_self_loops=False)
        # self.up_conv2 = GATConv(hidden_size, hidden_size)
        self.down_conv1 = GATConv(
            state_size, hidden_size, add_self_loops=False)
        # self.down_conv2 = GATConv(hidden_size, hidden_size)

        # self.up_conv1 = GCNConv(state_size, hidden_size)
        # self.up_conv2 = GCNConv(hidden_size, hidden_size)
        # self.down_conv1 = GCNConv(state_size, hidden_size)
        # self.down_conv2 = GCNConv(hidden_size, hidden_size)

        self.mlp = torch.nn.Linear(hidden_size, 1)
        # if init_type == 'normal':
        #     torch.nn.init.normal_(self.mlp.weight)

    def forward(self, batch_up_data, batch_down_data):
        up_x, up_edge_index, up_batch = batch_up_data.x, batch_up_data.edge_index, batch_up_data.batch
        # up_x = up_x.to(torch.float32)
        up_x = self.up_conv1(up_x.float(), up_edge_index)

        # for upstream event graph
        # get the number of nodes in each graph in the batch
        up_num_nodes_per_graph = scatter_sum(up_batch.new_ones(
            batch_up_data.num_nodes), up_batch, dim=0)
        # compute the indices of the self node in each graph
        up_self_index = torch.cumsum(up_num_nodes_per_graph, dim=0) - 1
        # get the embedding of the self node in each graph
        up_self_node_embed = up_x[up_self_index]
        up_self_node_embed = F.sigmoid(up_self_node_embed)

        # for downstream event graph
        # same as above
        down_x, down_edge_index, down_batch = batch_down_data.x, batch_down_data.edge_index, batch_down_data.batch
        down_x = self.down_conv1(down_x.float(), down_edge_index)
        down_num_nodes_per_graph = scatter_sum(down_batch.new_ones(
            batch_down_data.num_nodes), down_batch, dim=0)
        down_self_index = torch.cumsum(down_num_nodes_per_graph, dim=0) - 1
        down_self_node_embed = down_x[down_self_index]
        down_self_node_embed = F.sigmoid(down_self_node_embed)

        # x = up_self_node_embed * down_self_node_embed
        x = up_self_node_embed + down_self_node_embed
        x = self.mlp(x)

        return x
