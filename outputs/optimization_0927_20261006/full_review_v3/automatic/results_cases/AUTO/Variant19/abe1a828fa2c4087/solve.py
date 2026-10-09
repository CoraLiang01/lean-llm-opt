import gurobipy as gp
import pandas as pd
import numpy as np
import re
nodes_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant19/inputs/network_nodes.csv', dtype=str, keep_default_na=False)
edges_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant19/inputs/network_edges.csv', dtype=str, keep_default_na=False)
params_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant19/inputs/network_parameters.csv', dtype=str, keep_default_na=False)
nodes = nodes_df['Node'].str.strip().tolist()
n = len(nodes)
if n < 2:
    raise ValueError('At least two nodes are required for a spanning tree.')
edges = []
edge_cost = {}
for (idx, row) in edges_df.iterrows():
    i = row['Node1'].strip()
    j = row['Node2'].strip()
    if i not in nodes or j not in nodes:
        raise ValueError(f'Edge ({i},{j}) references unknown node(s).')
    (a, b) = sorted([i, j])
    key = (a, b)
    if key in edge_cost:
        raise ValueError(f'Duplicate undirected edge ({a},{b}) in input.')
    edges.append(key)
    try:
        cost = int(row['ConstructionCost'])
    except Exception:
        raise ValueError(f"Invalid ConstructionCost for edge ({i},{j}): {row['ConstructionCost']}")
    edge_cost[key] = cost
arcs = []
for (i, j) in edges:
    arcs.append((i, j))
    arcs.append((j, i))
root_row = params_df[params_df['Parameter'].str.strip().str.casefold() == 'rootnode']
if root_row.empty:
    raise ValueError('RootNode parameter not found in network_parameters.csv')
root_node = root_row.iloc[0]['Value'].strip()
if root_node not in nodes:
    raise ValueError(f"Root node '{root_node}' not found in node list.")
M = n - 1
m = gp.Model('MinCostSpanningTree_SCF')
y_vars = m.addVars(edges, vtype=gp.GRB.BINARY, name='')
f_vars = m.addVars(arcs, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((edge_cost[e] * y_vars[e] for e in edges)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((y_vars[e] for e in edges)) == n - 1, name='EdgeCount')
m.addConstr(gp.quicksum((f_vars[root_node, j] for j in nodes if (root_node, j) in f_vars)) == n - 1, name='RootFlowBalance')
for v in nodes:
    if v == root_node:
        continue
    inflow = gp.quicksum((f_vars[i, v] for i in nodes if (i, v) in f_vars))
    outflow = gp.quicksum((f_vars[v, j] for j in nodes if (v, j) in f_vars))
    m.addConstr(inflow - outflow == 1, name=f'FlowBalance_{v}')
for (i, j) in arcs:
    (a, b) = sorted([i, j])
    undirected_key = (a, b)
    m.addConstr(f_vars[i, j] <= M * y_vars[undirected_key], name=f'FlowLink_{i}_{j}')
m.optimize()