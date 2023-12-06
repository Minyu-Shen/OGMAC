import numpy as np
from collections import deque
import torch
from .agent import Agent
from torch_geometric.data import Data, Batch


class Graph_Agent(Agent):
    def __init__(self, config, agent_config, is_eval) -> None:
        super(Graph_Agent, self).__init__(config, agent_config, is_eval)

    def construct_graph(self, graps):
        # there are 'batch_size' graphs, construct them
        up_graph_list = []
        down_graph_list = []
        # for each graph, construct edge_index and x
        # each graph is a list containing events (nodes in namaedtuple)
        for graph in graps:
            # edge_index = torch.tensor([[1, 0],[2, 0]], dtype=torch.long)
            # x = torch.tensor([[0.116, 0.161, 0.25], [0.126, 0.141, 0.15], [0.111, 0.121, 0.55]], dtype=torch.float32)
            up_edges, down_edges = [], []
            up_edge_count, down_edge_count = 0, 0
            up_hs, down_hs = [], []
            for event in graph:
                # TODO check self augme is 0
                if event.up_or_down == 'up':
                    up_edge_count += 1
                    # up_edges.append([0, up_edge_count])
                    up_edges.append([up_edge_count, 0])
                    up_h = []
                    up_h.extend(event.state)
                    up_h.append(event.action)
                    up_h.append(event.augme_info)
                    up_hs.append(up_h)
                elif event.up_or_down == 'down':
                    down_edge_count += 1
                    # down_edges.append([0, down_edge_count])
                    down_edges.append([down_edge_count, 0])
                    down_h = []
                    down_h.extend(event.state)
                    down_h.append(event.action)
                    down_h.append(event.augme_info)
                    down_hs.append(down_h)
                elif event.up_or_down == 'self':
                    up_edges.append([0, 0])
                    up_h = []
                    up_h.extend(event.state)
                    up_h.append(event.action)
                    up_h.append(event.augme_info)
                    up_hs.append(up_h)

                    down_edges.append([0, 0])
                    down_h = []
                    down_h.extend(event.state)
                    down_h.append(event.action)
                    down_h.append(event.augme_info)
                    down_hs.append(down_h)

            up_edge_index = torch.tensor(up_edges)
            up_x = torch.tensor(up_hs, dtype=torch.float32)

            down_edge_index = torch.tensor(down_edges)
            down_x = torch.tensor(down_hs, dtype=torch.float32)

            # must be transposed so that the first row is the source node and the second row is the target node
            up_data = Data(x=up_x, edge_index=up_edge_index.t().contiguous())
            down_data = Data(
                x=down_x, edge_index=down_edge_index.t().contiguous())

            up_graph_list.append(up_data)
            down_graph_list.append(down_data)

        return up_graph_list, down_graph_list
