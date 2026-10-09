import gurobipy as gp
import pandas as pd
import numpy as np
import re
nodes_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant19/inputs/network_nodes.csv', dtype=str, keep_default_na=False)
edges_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant19/inputs/network_edges.csv', dtype=str, keep_default_na=False)
params_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant19/inputs/network_parameters.csv', dtype=str, keep_default_na=False)
nodes = nodes_df['Node'].str.strip().tolist()
n_nodes = len(nodes)
node_set = set(nodes)
edges = []
edge_cost = {}
for (idx, row) in edges_df.iterrows():
    i = row['Node1'].strip()
    j = row['Node2'].strip()
    if i not in node_set or j not in node_set:
        raise ValueError(f'Edge ({i},{j}) contains node not in node set.')
    key = tuple(sorted([i, j]))
    if key in edge_cost:
        raise ValueError(f'Duplicate undirected edge: {key}')
    edges.append(key)
    try:
        cost = int(row['ConstructionCost'])
    except Exception:
        raise ValueError(f"Invalid ConstructionCost for edge ({i},{j}): {row['ConstructionCost']}")
    edge_cost[key] = cost
root_row = params_df[params_df['Parameter'].str.strip().str.casefold() == 'rootnode']
if root_row.empty:
    raise ValueError('RootNode parameter not found in network_parameters.csv')
root_node = root_row.iloc[0]['Value'].strip()
if root_node not in node_set:
    raise ValueError(f"Root node '{root_node}' not found in node set.")
arcs = []
for (i, j) in edges:
    arcs.append((i, j))
    arcs.append((j, i))
M = n_nodes - 1
m = gp.Model('MinCostSpanningTree_SCF')
y_vars = m.addVars(edges, vtype=gp.GRB.BINARY, name='')
f_vars = m.addVars(arcs, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((edge_cost[edge] * y_vars[edge] for edge in edges)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((y_vars[edge] for edge in edges)) == n_nodes - 1, name='EdgeCount')
m.addConstr(gp.quicksum((f_vars[root_node, j] for j in nodes if (root_node, j) in f_vars)) == n_nodes - 1, name='RootFlowBalance')
for v in nodes:
    if v == root_node:
        continue
    inflow = gp.quicksum((f_vars[u, v] for u in nodes if (u, v) in f_vars))
    outflow = gp.quicksum((f_vars[v, w] for w in nodes if (v, w) in f_vars))
    m.addConstr(inflow - outflow == 1, name=f'FlowBalance_{v}')
for (i, j) in arcs:
    undirected_key = tuple(sorted([i, j]))
    m.addConstr(f_vars[i, j] <= M * y_vars[undirected_key], name=f'FlowToEdge_{i}_{j}')
m.optimize()