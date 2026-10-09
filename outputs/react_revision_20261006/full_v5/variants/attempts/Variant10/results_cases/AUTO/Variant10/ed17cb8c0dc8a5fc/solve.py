import gurobipy as gp
import pandas as pd
import numpy as np
import re

def normalize_id(x):
    return str(x).strip()
nodes_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant10/inputs/network_nodes.csv', sep=',')
edges_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant10/inputs/network_edges.csv', sep=',')
params_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant10/inputs/network_parameters.csv', sep=',')
nodes_df['Node'] = nodes_df['Node'].apply(normalize_id)
edges_df['Node1'] = edges_df['Node1'].apply(normalize_id)
edges_df['Node2'] = edges_df['Node2'].apply(normalize_id)
params_df['Parameter'] = params_df['Parameter'].apply(normalize_id)
params_df['Value'] = params_df['Value'].apply(normalize_id)
nodes = list(nodes_df['Node'].unique())
n_nodes = len(nodes)
if n_nodes < 2:
    raise ValueError('At least two nodes are required for a spanning tree.')
node_set = set(nodes)
for (idx, row) in edges_df.iterrows():
    if row['Node1'] not in node_set or row['Node2'] not in node_set:
        raise ValueError(f"Edge ({row['Node1']}, {row['Node2']}) references unknown node.")
edges = []
edge_cost = {}
for (idx, row) in edges_df.iterrows():
    i = row['Node1']
    j = row['Node2']
    cost = row['ConstructionCost']
    key = tuple(sorted([i, j]))
    if key in edge_cost:
        raise ValueError(f'Duplicate edge between {i} and {j}.')
    edges.append(key)
    edge_cost[key] = cost
arcs = []
arc_to_edge = {}
for (i, j) in edges:
    arcs.append((i, j))
    arcs.append((j, i))
    arc_to_edge[i, j] = (i, j) if (i, j) in edge_cost else (j, i)
    arc_to_edge[j, i] = (i, j) if (i, j) in edge_cost else (j, i)
root_row = params_df[params_df['Parameter'].str.casefold() == 'rootnode']
if root_row.empty:
    raise ValueError('RootNode parameter not found in network_parameters.csv')
root_node = root_row.iloc[0]['Value']
root_node = normalize_id(root_node)
if root_node not in node_set:
    raise ValueError(f"Root node '{root_node}' not found in node list.")
m = gp.Model('MinimumSpanningTree_FlowConnectivity')
y = m.addVars(edges, vtype=gp.GRB.BINARY, name='')
f = m.addVars(arcs, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((edge_cost[e] * y[e] for e in edges)), gp.GRB.MINIMIZE)
m.addConstr(y.sum() == n_nodes - 1, name='TreeSize')
for v in nodes:
    inflow = gp.quicksum((f[i, v] for (i, j) in arcs if j == v))
    outflow = gp.quicksum((f[v, j] for (i, j) in arcs if i == v))
    if v == root_node:
        m.addConstr(outflow - inflow == n_nodes - 1, name=f'FlowRoot_{v}')
    else:
        m.addConstr(inflow - outflow == 1, name=f'FlowNode_{v}')
for (i, j) in arcs:
    e = tuple(sorted([i, j]))
    m.addConstr(f[i, j] <= (n_nodes - 1) * y[e], name=f'FlowCap_{i}_{j}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for e in edges:
        print(f'y[{e[0]},{e[1]}] {y[e].VarName} {y[e].X}')
    for (i, j) in arcs:
        print(f'f[{i},{j}] {f[i, j].VarName} {f[i, j].X}')
else:
    print(f'Solver status: {m.status}')