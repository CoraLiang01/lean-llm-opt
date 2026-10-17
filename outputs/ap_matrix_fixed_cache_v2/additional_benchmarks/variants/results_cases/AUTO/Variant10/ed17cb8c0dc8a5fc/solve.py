import gurobipy as gp
import pandas as pd
import numpy as np
import re
nodes_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant10/inputs/network_nodes.csv', dtype=str, keep_default_na=False)
edges_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant10/inputs/network_edges.csv', dtype=str, keep_default_na=False)
params_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant10/inputs/network_parameters.csv', dtype=str, keep_default_na=False)
nodes = nodes_df['Node'].str.strip().tolist()
n_nodes = len(nodes)
if n_nodes < 2:
    raise ValueError('At least two nodes are required for a spanning tree.')
edges = []
edge_cost = {}
for idx, row in edges_df.iterrows():
    i = row['Node1'].strip()
    j = row['Node2'].strip()
    if i == j:
        raise ValueError(f'Edge from node {i} to itself is not allowed.')
    try:
        cost = int(row['ConstructionCost'])
    except Exception:
        raise ValueError(f"Invalid ConstructionCost for edge ({i},{j}): {row['ConstructionCost']}")
    edge = tuple(sorted([i, j]))
    edges.append(edge)
    edge_cost[edge] = cost
edges = list(set(edges))
arcs = []
arc_to_edge = {}
for i, j in edges:
    arcs.append((i, j))
    arcs.append((j, i))
    arc_to_edge[i, j] = tuple(sorted([i, j]))
    arc_to_edge[j, i] = tuple(sorted([i, j]))
root_row = params_df[params_df['Parameter'].str.strip().str.casefold() == 'rootnode']
if root_row.empty:
    raise ValueError('RootNode parameter not found in network_parameters.csv')
root_node = root_row.iloc[0]['Value'].strip()
if root_node not in nodes:
    raise ValueError(f"Root node '{root_node}' not found in node list.")
m = gp.Model('MinimumSpanningTree_Flow')
y_vars = m.addVars(edges, vtype=gp.GRB.BINARY, name='')
f_vars = m.addVars(arcs, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((edge_cost[e] * y_vars[e] for e in edges)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((y_vars[e] for e in edges)) == n_nodes - 1, name='SpanningTreeSize')
for v in nodes:
    if v == root_node:
        m.addConstr(gp.quicksum((f_vars[root_node, j] for j in nodes if (root_node, j) in f_vars)) - gp.quicksum((f_vars[j, root_node] for j in nodes if (j, root_node) in f_vars)) == n_nodes - 1, name=f'FlowConservation_{root_node}')
    else:
        m.addConstr(gp.quicksum((f_vars[v, j] for j in nodes if (v, j) in f_vars)) - gp.quicksum((f_vars[j, v] for j in nodes if (j, v) in f_vars)) == -1, name=f'FlowConservation_{v}')
for i, j in arcs:
    e = arc_to_edge[i, j]
    m.addConstr(f_vars[i, j] <= (n_nodes - 1) * y_vars[e], name=f'FlowEdgeLink_{i}_{j}')
m.optimize()