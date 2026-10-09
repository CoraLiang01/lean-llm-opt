import gurobipy as gp
import pandas as pd
import numpy as np
import re

def norm_id(x):
    return str(x).strip()
nodes_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant19/inputs/network_nodes.csv', dtype=str, keep_default_na=False)
edges_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant19/inputs/network_edges.csv', dtype=str, keep_default_na=False)
params_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant19/inputs/network_parameters.csv', dtype=str, keep_default_na=False)
nodes = [norm_id(n) for n in nodes_df['Node']]
n_nodes = len(nodes)
node_set = set(nodes)
edges = []
edge_cost = {}
for (idx, row) in edges_df.iterrows():
    i = norm_id(row['Node1'])
    j = norm_id(row['Node2'])
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
    except Exception:
        raise ValueError(f"Invalid ConstructionCost for edge ({i},{j}): {row['ConstructionCost']}")
    edge_cost[edge] = cost
arcs = []
for (i, j) in edges:
    arcs.append((i, j))
    arcs.append((j, i))
root_row = params_df[params_df['Parameter'].str.strip().str.casefold() == 'rootnode']
if root_row.empty:
    raise ValueError('RootNode parameter not found in network_parameters.csv')
root_node = norm_id(root_row.iloc[0]['Value'])
if root_node not in node_set:
    raise ValueError(f"Root node '{root_node}' not found in node set.")
M = n_nodes - 1
m = gp.Model('MinCostSpanningTree_SCF')
y_vars = m.addVars(edges, vtype=gp.GRB.BINARY, name='')
f_vars = m.addVars(arcs, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((edge_cost[edge] * y_vars[edge] for edge in edges)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((y_vars[edge] for edge in edges)) == n_nodes - 1, name='EdgeCount')
m.addConstr(gp.quicksum((f_vars[root_node, j] for j in nodes if (root_node, j) in f_vars)) == n_nodes - 1, name='RootFlowBalance')
for k in nodes:
    if k == root_node:
        continue
    arcs_in = [(i, k) for i in nodes if (i, k) in f_vars]
    arcs_out = [(k, j) for j in nodes if (k, j) in f_vars]
    m.addConstr(gp.quicksum((f_vars[a] for a in arcs_in)) - gp.quicksum((f_vars[a] for a in arcs_out)) == 1, name=f'FlowBalance_{k}')
for (i, j) in arcs:
    undirected = (i, j) if i < j else (j, i)
    m.addConstr(f_vars[i, j] <= M * y_vars[undirected], name=f'FlowLink_{i}_{j}')
m.optimize()