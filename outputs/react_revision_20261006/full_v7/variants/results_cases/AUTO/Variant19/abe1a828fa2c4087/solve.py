import gurobipy as gp
import pandas as pd
import numpy as np
import re

def normalize_node_id(x):
    return str(x).strip()
nodes_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant19/inputs/network_nodes.csv', dtype=str, keep_default_na=False)
edges_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant19/inputs/network_edges.csv', dtype=str, keep_default_na=False)
params_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant19/inputs/network_parameters.csv', dtype=str, keep_default_na=False)
nodes_df['Node'] = nodes_df['Node'].apply(normalize_node_id)
edges_df['Node1'] = edges_df['Node1'].apply(normalize_node_id)
edges_df['Node2'] = edges_df['Node2'].apply(normalize_node_id)
N = list(nodes_df['Node'])
n = len(N)
if n < 2:
    raise ValueError('At least two nodes are required for a spanning tree.')
E = []
edge_cost = {}
for (idx, row) in edges_df.iterrows():
    i = row['Node1']
    j = row['Node2']
    if i not in N or j not in N:
        raise ValueError(f'Edge ({i},{j}) contains node not in node set.')
    key = tuple(sorted((i, j)))
    if key in edge_cost:
        raise ValueError(f'Duplicate undirected edge between {i} and {j}.')
    E.append(key)
    try:
        cost = int(row['ConstructionCost'])
    except Exception:
        raise ValueError(f"Invalid ConstructionCost for edge ({i},{j}): {row['ConstructionCost']}")
    edge_cost[key] = cost
A = []
arc_to_edge = {}
for (i, j) in E:
    A.append((i, j))
    A.append((j, i))
    arc_to_edge[i, j] = (min(i, j), max(i, j))
    arc_to_edge[j, i] = (min(i, j), max(i, j))
root_row = params_df[params_df['Parameter'].str.casefold().str.strip() == 'rootnode']
if root_row.empty:
    raise ValueError('RootNode parameter not found in network_parameters.csv')
root_node = normalize_node_id(root_row.iloc[0]['Value'])
if root_node not in N:
    raise ValueError(f"Root node '{root_node}' not found in node set.")
M = n - 1
m = gp.Model('MST_SingleCommodityFlow')
m.Params.MIPGap = 0.0001
y_keys = E
y_vars = m.addVars(y_keys, vtype=gp.GRB.BINARY, name='')
f_keys = A
f_vars = m.addVars(f_keys, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((edge_cost[i, j] * y_vars[i, j] for (i, j) in E)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((y_vars[i, j] for (i, j) in E)) == n - 1, name='edge_count')
out_arcs = [(root_node, j) for j in N if j != root_node and (root_node, j) in f_vars]
in_arcs = [(j, root_node) for j in N if j != root_node and (j, root_node) in f_vars]
m.addConstr(gp.quicksum((f_vars[a] for a in out_arcs)) - gp.quicksum((f_vars[a] for a in in_arcs)) == n - 1, name='root_flow_balance')
for k in N:
    if k == root_node:
        continue
    in_arcs_k = [(i, k) for i in N if i != k and (i, k) in f_vars]
    out_arcs_k = [(k, j) for j in N if j != k and (k, j) in f_vars]
    m.addConstr(gp.quicksum((f_vars[a] for a in in_arcs_k)) - gp.quicksum((f_vars[a] for a in out_arcs_k)) == 1, name=f'flow_balance_{k}')
for (i, j) in A:
    undirected = arc_to_edge[i, j]
    m.addConstr(f_vars[i, j] <= M * y_vars[undirected], name=f'flow_link_{i}_{j}')
m.optimize()
if m.Status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.Status}')