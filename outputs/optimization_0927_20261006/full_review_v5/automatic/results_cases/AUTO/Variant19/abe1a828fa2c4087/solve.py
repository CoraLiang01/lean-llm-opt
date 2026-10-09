import gurobipy as gp
import pandas as pd
import numpy as np
import re
nodes_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant19/inputs/network_nodes.csv', dtype=str, keep_default_na=False)
edges_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant19/inputs/network_edges.csv', dtype=str, keep_default_na=False)
params_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant19/inputs/network_parameters.csv', dtype=str, keep_default_na=False)
nodes = nodes_df['Node'].str.strip().tolist()
n_nodes = len(nodes)
if len(set(nodes)) != n_nodes:
    raise ValueError('Duplicate node identifiers found in network_nodes.csv.')
root_row = params_df[params_df['Parameter'].str.strip().str.casefold() == 'rootnode']
if root_row.empty:
    raise ValueError('RootNode parameter not found in network_parameters.csv.')
root_node = root_row.iloc[0]['Value'].strip()
if root_node not in nodes:
    raise ValueError(f"Root node '{root_node}' not found in node list.")
edges_df['Node1'] = edges_df['Node1'].str.strip()
edges_df['Node2'] = edges_df['Node2'].str.strip()
edges_df['ConstructionCost'] = edges_df['ConstructionCost'].astype(float)
edges_df = edges_df[edges_df['Node1'].isin(nodes) & edges_df['Node2'].isin(nodes)].copy()

def undirected_edge_key(row):
    (i, j) = (row['Node1'], row['Node2'])
    return tuple(sorted((i, j)))
edges_df['edge_key'] = edges_df.apply(undirected_edge_key, axis=1)
edges_df = edges_df.drop_duplicates(subset=['edge_key'])
undirected_edges = edges_df['edge_key'].tolist()
edge_cost = dict(zip(edges_df['edge_key'], edges_df['ConstructionCost']))
directed_arcs = []
for (i, j) in undirected_edges:
    directed_arcs.append((i, j))
    directed_arcs.append((j, i))
from collections import defaultdict
out_arcs = defaultdict(list)
in_arcs = defaultdict(list)
for (u, v) in directed_arcs:
    out_arcs[u].append((u, v))
    in_arcs[v].append((u, v))
m = gp.Model('MinCostSpanningTree_Flow')
y_vars = m.addVars(undirected_edges, vtype=gp.GRB.BINARY, name='')
f_vars = m.addVars(directed_arcs, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((edge_cost[e] * y_vars[e] for e in undirected_edges)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((y_vars[e] for e in undirected_edges)) == n_nodes - 1, name='EdgeCount')
m.addConstr(gp.quicksum((f_vars[arc] for arc in out_arcs[root_node])) == n_nodes - 1, name='RootFlowBalance')
for k in nodes:
    if k == root_node:
        continue
    m.addConstr(gp.quicksum((f_vars[arc] for arc in in_arcs[k])) - gp.quicksum((f_vars[arc] for arc in out_arcs[k])) == 1, name=f'FlowBalance_{k}')
for (i, j) in directed_arcs:
    edge_key = tuple(sorted((i, j)))
    m.addConstr(f_vars[i, j] <= (n_nodes - 1) * y_vars[edge_key], name=f'FlowLink_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total construction cost: {m.objVal:.2f}')
    print('\n--- Selected Edges (y_ij = 1) ---')
    for e in undirected_edges:
        if y_vars[e].X > 0.5:
            print(f'  Edge {e[0]} - {e[1]} (Cost: {edge_cost[e]:.2f})')
    print('\n--- Flow on Directed Arcs (f_ij > 0) ---')
    for arc in directed_arcs:
        if f_vars[arc].X > 1e-06:
            print(f'  Flow {arc[0]} -> {arc[1]}: {f_vars[arc].X:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')