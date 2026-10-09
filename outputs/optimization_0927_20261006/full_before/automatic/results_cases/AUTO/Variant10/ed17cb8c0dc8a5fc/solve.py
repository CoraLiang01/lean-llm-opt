import gurobipy as gp
import pandas as pd
import numpy as np
import re

def find_col(df, pattern):
    for col in df.columns:
        if re.search(pattern, col, re.IGNORECASE):
            return col
    raise KeyError(f"Could not find a column matching pattern '{pattern}'")
nodes_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant10/inputs/network_nodes.csv', sep=',')
edges_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant10/inputs/network_edges.csv', sep=',')
params_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant10/inputs/network_parameters.csv', sep=',')
nodes = nodes_df['Node'].astype(str).str.strip().tolist()
n_nodes = len(nodes)
if n_nodes < 2:
    raise ValueError('At least two nodes are required for a spanning tree.')
edges = []
edge_cost = {}
for (idx, row) in edges_df.iterrows():
    i = str(row['Node1']).strip()
    j = str(row['Node2']).strip()
    if i == j:
        raise ValueError(f'Self-loop detected in edge ({i},{j})')
    e = tuple(sorted([i, j]))
    if e in edge_cost:
        raise ValueError(f'Duplicate undirected edge between {i} and {j}')
    edges.append(e)
    edge_cost[e] = float(row['ConstructionCost'])
arcs = []
for (i, j) in edges:
    arcs.append((i, j))
    arcs.append((j, i))
root_row = params_df[params_df['Parameter'].str.strip().str.casefold() == 'rootnode']
if root_row.empty:
    raise ValueError('RootNode parameter not found in network_parameters.csv')
root_node = str(root_row.iloc[0]['Value']).strip()
if root_node not in nodes:
    raise ValueError(f"Root node '{root_node}' not found in node list.")
m = gp.Model('MinimumSpanningTree_FlowBased')
y = m.addVars(edges, vtype=gp.GRB.BINARY, name='')
f = m.addVars(arcs, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((edge_cost[e] * y[e] for e in edges)), gp.GRB.MINIMIZE)
m.addConstr(y.sum() == n_nodes - 1, name='SpanningTreeCardinality')
for v in nodes:
    inflow = gp.quicksum((f[i, v] for (i, j) in arcs if j == v))
    outflow = gp.quicksum((f[v, j] for (i, j) in arcs if i == v))
    if v == root_node:
        m.addConstr(outflow - inflow == n_nodes - 1, name=f'FlowConsv_{v}')
    else:
        m.addConstr(inflow - outflow == 1, name=f'FlowConsv_{v}')
for (i, j) in arcs:
    e = tuple(sorted([i, j]))
    m.addConstr(f[i, j] <= (n_nodes - 1) * y[e], name=f'FlowLink_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total construction cost: {m.objVal:.2f}')
    print('\n--- Selected Links (edges) ---')
    for e in edges:
        if y[e].X > 0.5:
            print(f'  Link: {e[0]} -- {e[1]} (Cost: {edge_cost[e]})')
    print('\n--- Nonzero Flows (f_ij) ---')
    for (i, j) in arcs:
        if f[i, j].X > 1e-06:
            print(f'  Flow from {i} to {j}: {f[i, j].X:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')