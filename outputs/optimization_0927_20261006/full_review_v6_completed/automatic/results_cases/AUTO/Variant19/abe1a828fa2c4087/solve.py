import gurobipy as gp
import pandas as pd
import numpy as np
import re
nodes_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant19/inputs/network_nodes.csv', dtype=str, keep_default_na=False)
edges_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant19/inputs/network_edges.csv', dtype=str, keep_default_na=False)
params_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant19/inputs/network_parameters.csv', dtype=str, keep_default_na=False)
nodes = nodes_df['Node'].str.strip().tolist()
n = len(nodes)
node_set = set(nodes)
edges = []
edge_cost = {}
for (idx, row) in edges_df.iterrows():
    i = row['Node1'].strip()
    j = row['Node2'].strip()
    if i not in node_set or j not in node_set:
        raise ValueError(f'Edge ({i},{j}) contains node not in node set.')
    if i < j:
        edge = (i, j)
    else:
        edge = (j, i)
    if edge in edge_cost:
        raise ValueError(f'Duplicate undirected edge: {edge}')
    edges.append(edge)
    try:
        cost = int(row['ConstructionCost'])
    except Exception as e:
        raise ValueError(f"Invalid ConstructionCost for edge {edge}: {row['ConstructionCost']}")
    edge_cost[edge] = cost
arcs = []
for (i, j) in edges:
    arcs.append((i, j))
    arcs.append((j, i))
root_row = params_df[params_df['Parameter'].str.strip().str.casefold() == 'rootnode']
if root_row.empty:
    raise ValueError('RootNode parameter not found in network_parameters.csv')
root_node = root_row.iloc[0]['Value'].strip()
if root_node not in node_set:
    raise ValueError(f"Root node '{root_node}' not found in node set.")
M = n - 1
m = gp.Model('min_cost_spanning_tree_flow')
y_vars = m.addVars(edges, vtype=gp.GRB.BINARY, name='')
f_vars = m.addVars(arcs, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((edge_cost[edge] * y_vars[edge] for edge in edges)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((y_vars[edge] for edge in edges)) == n - 1, name='edge_count')
m.addConstr(gp.quicksum((f_vars[root_node, j] for j in nodes if (root_node, j) in f_vars)) == n - 1, name='root_flow_balance')
for k in nodes:
    if k == root_node:
        continue
    m.addConstr(gp.quicksum((f_vars[i, k] for i in nodes if (i, k) in f_vars)) - gp.quicksum((f_vars[k, j] for j in nodes if (k, j) in f_vars)) == 1, name=f'flow_balance_{k}')
for (i, j) in arcs:
    edge = (i, j) if i < j else (j, i)
    m.addConstr(f_vars[i, j] <= M * y_vars[edge], name=f'flow_link_{i}_{j}')
m.optimize()