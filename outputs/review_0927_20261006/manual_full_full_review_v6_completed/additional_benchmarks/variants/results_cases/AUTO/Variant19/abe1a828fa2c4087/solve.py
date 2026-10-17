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
edges_df['Node1'] = edges_df['Node1'].str.strip()
edges_df['Node2'] = edges_df['Node2'].str.strip()
edges = []
edge_cost = {}
for idx, row in edges_df.iterrows():
    i = row['Node1']
    j = row['Node2']
    edge_key = tuple(sorted([i, j]))
    edges.append(edge_key)
    edge_cost[edge_key] = int(row['ConstructionCost'])
arcs = []
for i, j in edges:
    arcs.append((i, j))
    arcs.append((j, i))

def undirected_key(i, j):
    return tuple(sorted([i, j]))
params_df['Parameter'] = params_df['Parameter'].str.strip().str.casefold()
root_row = params_df[params_df['Parameter'] == 'rootnode']
if root_row.empty:
    raise ValueError('RootNode parameter not found in network_parameters.csv')
root_node = root_row.iloc[0]['Value'].strip()
if root_node not in node_set:
    raise ValueError(f"Root node '{root_node}' not found in node set.")
M = n - 1
m = gp.Model('MinCostSpanningTree_SCF')
y_vars = m.addVars(edges, vtype=gp.GRB.BINARY, name='')
f_vars = m.addVars(arcs, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((edge_cost[edge] * y_vars[edge] for edge in edges)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((y_vars[edge] for edge in edges)) == n - 1, name='EdgeCount')
m.addConstr(gp.quicksum((f_vars[root_node, j] for j in nodes if (root_node, j) in f_vars)) == n - 1, name='RootFlowBalance')
for k in nodes:
    if k == root_node:
        continue
    inflow = gp.quicksum((f_vars[i, k] for i in nodes if (i, k) in f_vars))
    outflow = gp.quicksum((f_vars[k, j] for j in nodes if (k, j) in f_vars))
    m.addConstr(inflow - outflow == 1, name=f'FlowBalance_{k}')
for i, j in arcs:
    m.addConstr(f_vars[i, j] <= M * y_vars[undirected_key(i, j)], name=f'FlowToEdge_{i}_{j}')
m.optimize()