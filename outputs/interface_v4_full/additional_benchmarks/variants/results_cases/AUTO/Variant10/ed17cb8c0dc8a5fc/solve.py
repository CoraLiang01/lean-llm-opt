import gurobipy as gp
import pandas as pd
import numpy as np
import re

def normalize_id(x):
    return re.sub('\\s+', '', str(x))
nodes_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant10/inputs/network_nodes.csv', sep=',')
edges_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant10/inputs/network_edges.csv', sep=',')
params_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant10/inputs/network_parameters.csv', sep=',')
nodes_df['Node'] = nodes_df['Node'].apply(normalize_id)
edges_df['Node1'] = edges_df['Node1'].apply(normalize_id)
edges_df['Node2'] = edges_df['Node2'].apply(normalize_id)
params_df['Parameter'] = params_df['Parameter'].apply(lambda x: re.sub('\\s+', '', str(x)))
params_df['Value'] = params_df['Value'].apply(normalize_id)
nodes = list(nodes_df['Node'])
n_nodes = len(nodes)
edges = []
edge_cost = {}
for idx, row in edges_df.iterrows():
    i = row['Node1']
    j = row['Node2']
    cost = row['ConstructionCost']
    key = tuple(sorted([i, j]))
    edges.append(key)
    edge_cost[key] = cost
arcs = []
for i, j in edges:
    arcs.append((i, j))
    arcs.append((j, i))
root_row = params_df[params_df['Parameter'] == 'RootNode']
if root_row.empty:
    raise ValueError('RootNode parameter not found in network_parameters.csv')
root_node = root_row.iloc[0]['Value']
if root_node not in nodes:
    raise ValueError(f"Root node '{root_node}' not found in node list.")
m = gp.Model('MinimumSpanningTree_FlowConnectivity')
y = m.addVars(edges, vtype=gp.GRB.BINARY, name='')
f = m.addVars(arcs, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((edge_cost[e] * y[e] for e in edges)), gp.GRB.MINIMIZE)
m.addConstr(y.sum() == n_nodes - 1, name='LinkCount')
for k in nodes:
    if k == root_node:
        m.addConstr(gp.quicksum((f[root_node, j] for j in nodes if (root_node, j) in arcs)) - gp.quicksum((f[i, root_node] for i in nodes if (i, root_node) in arcs)) == n_nodes - 1, name=f'FlowConserv_root')
    else:
        m.addConstr(gp.quicksum((f[i, k] for i in nodes if (i, k) in arcs)) - gp.quicksum((f[k, j] for j in nodes if (k, j) in arcs)) == 1, name=f'FlowConserv_{k}')
for i, j in arcs:
    e = tuple(sorted([i, j]))
    m.addConstr(f[i, j] <= (n_nodes - 1) * y[e], name=f'FlowLink_{i}_{j}')
m.optimize()