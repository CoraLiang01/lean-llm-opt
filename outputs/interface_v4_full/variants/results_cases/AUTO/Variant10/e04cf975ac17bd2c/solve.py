import gurobipy as gp
import pandas as pd
import numpy as np
nodes_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant10/inputs/network_nodes.csv', sep=',')
edges_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant10/inputs/network_edges.csv', sep=',')
params_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant10/inputs/network_parameters.csv', sep=',')

def norm_node(x):
    return str(x).strip()
nodes = [norm_node(n) for n in nodes_df['Node']]
n_nodes = len(nodes)
node_set = set(nodes)
edges = []
edge_cost = {}
for idx, row in edges_df.iterrows():
    i = norm_node(row['Node1'])
    j = norm_node(row['Node2'])
    if i not in node_set or j not in node_set:
        raise ValueError(f'Edge ({i},{j}) contains node not in node set')
    key = tuple(sorted([i, j]))
    edges.append(key)
    edge_cost[key] = float(row['ConstructionCost'])
root_row = params_df[params_df['Parameter'].str.strip().str.casefold() == 'rootnode']
if root_row.empty:
    raise ValueError('RootNode parameter not found in network_parameters.csv')
root_node = norm_node(root_row.iloc[0]['Value'])
if root_node not in node_set:
    raise ValueError(f'Root node {root_node} not in node set')
arcs = []
for i, j in edges:
    arcs.append((i, j))
    arcs.append((j, i))
arc_to_edge = {}
for i, j in edges:
    arc_to_edge[i, j] = tuple(sorted([i, j]))
    arc_to_edge[j, i] = tuple(sorted([i, j]))
m = gp.Model('MinimumSpanningTree_FlowConnectivity')
y = m.addVars(edges, vtype=gp.GRB.BINARY, name='')
f = m.addVars(arcs, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((edge_cost[e] * y[e] for e in edges)), gp.GRB.MINIMIZE)
m.addConstr(y.sum() == n_nodes - 1, name='SpanningTreeCardinality')
for k in nodes:
    if k == root_node:
        m.addConstr(gp.quicksum((f[root_node, j] for j in nodes if (root_node, j) in f)) - gp.quicksum((f[j, root_node] for j in nodes if (j, root_node) in f)) == n_nodes - 1, name=f'FlowConservation_root')
    else:
        m.addConstr(gp.quicksum((f[j, k] for j in nodes if (j, k) in f)) - gp.quicksum((f[k, j] for j in nodes if (k, j) in f)) == 1, name=f'FlowConservation_{k}')
for arc in arcs:
    undirected = arc_to_edge[arc]
    m.addConstr(f[arc] <= (n_nodes - 1) * y[undirected], name=f'FlowLink_{arc[0]}_{arc[1]}')
m.optimize()