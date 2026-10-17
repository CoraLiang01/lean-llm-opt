import gurobipy as gp
import pandas as pd
import numpy as np
import re

def normalize_id(x):
    return str(x).strip()
nodes_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant10/inputs/network_nodes.csv', dtype=str, keep_default_na=False)
edges_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant10/inputs/network_edges.csv', dtype=str, keep_default_na=False)
params_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant10/inputs/network_parameters.csv', dtype=str, keep_default_na=False)
nodes = [normalize_id(n) for n in nodes_df['Node']]
n_nodes = len(nodes)
node_set = set(nodes)
edges = []
edge_cost = dict()
for idx, row in edges_df.iterrows():
    i = normalize_id(row['Node1'])
    j = normalize_id(row['Node2'])
    if i == j:
        raise ValueError(f'Edge with identical nodes: {i}')
    ni, nj = sorted([i, j])
    edges.append((ni, nj))
    try:
        cost = int(row['ConstructionCost'])
    except Exception:
        raise ValueError(f"Invalid ConstructionCost for edge ({ni},{nj}): {row['ConstructionCost']}")
    edge_cost[ni, nj] = cost
for i, j in edges:
    if i not in node_set or j not in node_set:
        raise ValueError(f'Edge ({i},{j}) references node(s) not in node set.')
arcs = []
arc_to_edge = dict()
for i, j in edges:
    arcs.append((i, j))
    arcs.append((j, i))
    arc_to_edge[i, j] = (i, j)
    arc_to_edge[j, i] = (i, j)
root_row = params_df[params_df['Parameter'].str.strip().str.casefold() == 'rootnode']
if root_row.empty:
    raise ValueError('RootNode parameter not found in network_parameters.csv')
root_node = normalize_id(root_row.iloc[0]['Value'])
if root_node not in node_set:
    raise ValueError(f"Root node '{root_node}' not found in node set.")
m = gp.Model('MinimumSpanningTree_FlowConnectivity')
y_vars = m.addVars(edges, vtype=gp.GRB.BINARY, name='')
f_vars = m.addVars(arcs, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((edge_cost[e] * y_vars[e] for e in edges)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((y_vars[e] for e in edges)) == n_nodes - 1, name='SpanningTreeSize')
for k in nodes:
    if k == root_node:
        m.addConstr(gp.quicksum((f_vars[root_node, j] for j in nodes if (root_node, j) in arcs)) - gp.quicksum((f_vars[j, root_node] for j in nodes if (j, root_node) in arcs)) == n_nodes - 1, name=f'FlowConservation_root')
    else:
        m.addConstr(gp.quicksum((f_vars[i, k] for i in nodes if (i, k) in arcs)) - gp.quicksum((f_vars[k, j] for j in nodes if (k, j) in arcs)) == 1, name=f'FlowConservation_{k}')
for i, j in arcs:
    e = tuple(sorted([i, j]))
    m.addConstr(f_vars[i, j] <= (n_nodes - 1) * y_vars[e], name=f'FlowEdgeLink_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total construction cost: {m.objVal:.2f}')
    print('\n--- Selected Edges (y_e = 1) ---')
    for e in edges:
        if y_vars[e].X > 0.5:
            print(f'  Edge {e[0]} -- {e[1]} (Cost: {edge_cost[e]})')
    print('\n--- Nonzero Flows (f_ij > 1e-6) ---')
    for i, j in arcs:
        if f_vars[i, j].X > 1e-06:
            print(f'  Flow from {i} to {j}: {f_vars[i, j].X:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')