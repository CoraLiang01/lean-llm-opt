import gurobipy as gp
import pandas as pd
import numpy as np
import re

def normalize_id(x):
    return str(x).strip()
nodes_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant19/inputs/network_nodes.csv', sep=',')
edges_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant19/inputs/network_edges.csv', sep=',')
params_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant19/inputs/network_parameters.csv', sep=',')
nodes_df['Node'] = nodes_df['Node'].apply(normalize_id)
edges_df['Node1'] = edges_df['Node1'].apply(normalize_id)
edges_df['Node2'] = edges_df['Node2'].apply(normalize_id)
params_df['Parameter'] = params_df['Parameter'].apply(normalize_id)
params_df['Value'] = params_df['Value'].apply(normalize_id)
nodes = list(nodes_df['Node'])
n = len(nodes)
if n < 2:
    raise ValueError('At least two nodes are required for a spanning tree.')
edges = []
edge_cost = {}
for idx, row in edges_df.iterrows():
    i = row['Node1']
    j = row['Node2']
    cost = row['ConstructionCost']
    ni, nj = sorted([i, j])
    edges.append((ni, nj))
    edge_cost[ni, nj] = cost
for i, j in edges:
    if i not in nodes or j not in nodes:
        raise ValueError(f'Edge ({i},{j}) references node(s) not in node set.')
root_row = params_df[params_df['Parameter'].str.casefold() == 'rootnode']
if root_row.empty:
    raise ValueError('RootNode parameter not found in network_parameters.csv')
root = normalize_id(root_row.iloc[0]['Value'])
if root not in nodes:
    raise ValueError(f"Root node '{root}' not found in node set.")
arcs = []
for i, j in edges:
    arcs.append((i, j))
    arcs.append((j, i))
m = gp.Model('MinCostSpanningTree_SCF')
y = m.addVars(edges, vtype=gp.GRB.BINARY, name='')
f = m.addVars(arcs, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((edge_cost[e] * y[e] for e in edges)), gp.GRB.MINIMIZE)
m.addConstr(y.sum() == n - 1, name='EdgeCount')
m.addConstr(gp.quicksum((f[root, j] for j in nodes if (root, j) in arcs)) - gp.quicksum((f[j, root] for j in nodes if (j, root) in arcs)) == n - 1, name='RootFlowBalance')
for k in nodes:
    if k == root:
        continue
    m.addConstr(gp.quicksum((f[j, k] for j in nodes if (j, k) in arcs)) - gp.quicksum((f[k, j] for j in nodes if (k, j) in arcs)) == 1, name=f'NodeFlowBalance_{k}')
M = n - 1
for i, j in arcs:
    e = tuple(sorted([i, j]))
    m.addConstr(f[i, j] <= M * y[e], name=f'FlowLink_{i}_{j}')
m.optimize()