import gurobipy as gp
import pandas as pd
import numpy as np
import re
nodes_df = pd.read_csv('benchmark_dataset/Variant19/inputs/network_nodes.csv', sep=',', dtype=str, keep_default_na=False)
nodes_df['Node'] = nodes_df['Node'].str.strip()
nodes = list(nodes_df['Node'])
n_nodes = len(nodes)
if n_nodes < 2:
    raise ValueError('At least two nodes are required for a spanning tree.')
edges_df = pd.read_csv('benchmark_dataset/Variant19/inputs/network_edges.csv', sep=',', dtype=str, keep_default_na=False)
edges_df['Node1'] = edges_df['Node1'].str.strip()
edges_df['Node2'] = edges_df['Node2'].str.strip()
edges_df['ConstructionCost'] = edges_df['ConstructionCost'].astype(float)
for (idx, row) in edges_df.iterrows():
    if row['Node1'] not in nodes or row['Node2'] not in nodes:
        raise ValueError(f"Edge endpoints {row['Node1']}, {row['Node2']} not in node set.")
undirected_edges = []
edge_cost = {}
for (idx, row) in edges_df.iterrows():
    (i, j) = (row['Node1'], row['Node2'])
    key = tuple(sorted((i, j)))
    if key in edge_cost:
        raise ValueError(f'Duplicate undirected edge between {i} and {j}.')
    undirected_edges.append(key)
    edge_cost[key] = row['ConstructionCost']
directed_arcs = []
for (i, j) in undirected_edges:
    directed_arcs.append((i, j))
    directed_arcs.append((j, i))
params_df = pd.read_csv('benchmark_dataset/Variant19/inputs/network_parameters.csv', sep=',', dtype=str, keep_default_na=False)
params_df['Parameter'] = params_df['Parameter'].str.strip().str.casefold()
params_df['Value'] = params_df['Value'].str.strip()
root_row = params_df[params_df['Parameter'] == 'rootnode']
if root_row.empty:
    raise ValueError('RootNode parameter not found in network_parameters.csv.')
root_node = root_row.iloc[0]['Value']
if root_node not in nodes:
    raise ValueError(f'Root node {root_node} not found in node set.')
m = gp.Model('MinimumCostSpanningTree_SCF')
y_vars = m.addVars(undirected_edges, vtype=gp.GRB.BINARY, name='')
f_vars = m.addVars(directed_arcs, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((edge_cost[e] * y_vars[e] for e in undirected_edges)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((y_vars[e] for e in undirected_edges)) == n_nodes - 1, name='EdgeCount')
m.addConstr(gp.quicksum((f_vars[root_node, j] for j in nodes if (root_node, j) in f_vars)) == n_nodes - 1, name='RootFlowOut')
for k in nodes:
    if k == root_node:
        continue
    arcs_in = [(i, k) for i in nodes if (i, k) in f_vars]
    arcs_out = [(k, j) for j in nodes if (k, j) in f_vars]
    m.addConstr(gp.quicksum((f_vars[a] for a in arcs_in)) - gp.quicksum((f_vars[a] for a in arcs_out)) == 1, name=f'FlowBalance_{k}')
M = n_nodes - 1
for (i, j) in directed_arcs:
    undirected_key = tuple(sorted((i, j)))
    m.addConstr(f_vars[i, j] <= M * y_vars[undirected_key], name=f'FlowLink_{i}_{j}')
m.optimize()