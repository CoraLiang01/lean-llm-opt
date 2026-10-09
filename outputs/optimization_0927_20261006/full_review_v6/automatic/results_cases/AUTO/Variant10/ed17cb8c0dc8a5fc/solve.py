import gurobipy as gp
import pandas as pd
import numpy as np
import re

def normalize_id(x):
    return x.strip()
nodes_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant10/inputs/network_nodes.csv', dtype=str, keep_default_na=False)
edges_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant10/inputs/network_edges.csv', dtype=str, keep_default_na=False)
params_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant10/inputs/network_parameters.csv', dtype=str, keep_default_na=False)
nodes = [normalize_id(n) for n in nodes_df['Node']]
n_nodes = len(nodes)
node_set = set(nodes)
edges = []
edge_cost = {}
for (idx, row) in edges_df.iterrows():
    n1 = normalize_id(row['Node1'])
    n2 = normalize_id(row['Node2'])
    if n1 not in node_set or n2 not in node_set:
        raise ValueError(f'Edge ({n1},{n2}) references unknown node(s).')
    edge = (n1, n2) if n1 <= n2 else (n2, n1)
    if edge in edge_cost:
        raise ValueError(f'Duplicate edge ({edge[0]},{edge[1]}) in input.')
    edges.append(edge)
    try:
        cost = int(row['ConstructionCost'])
    except Exception:
        raise ValueError(f"Invalid ConstructionCost for edge ({n1},{n2}): {row['ConstructionCost']}")
    edge_cost[edge] = cost
root_row = params_df[params_df['Parameter'].str.strip().str.casefold() == 'rootnode']
if root_row.empty:
    raise ValueError('RootNode parameter not found in network_parameters.csv')
root_node = normalize_id(root_row.iloc[0]['Value'])
if root_node not in node_set:
    raise ValueError(f"Root node '{root_node}' not found in node set.")
arcs = []
arc_to_edge = {}
for (i, j) in edges:
    arcs.append((i, j))
    arcs.append((j, i))
    arc_to_edge[i, j] = (i, j) if (i, j) in edge_cost else (j, i)
    arc_to_edge[j, i] = (i, j) if (i, j) in edge_cost else (j, i)
m = gp.Model('MinimumSpanningTree_FlowConnectivity')
y_vars = m.addVars(edges, vtype=gp.GRB.BINARY, name='')
f_vars = m.addVars(arcs, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((edge_cost[e] * y_vars[e] for e in edges)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((y_vars[e] for e in edges)) == n_nodes - 1, name='SpanningTreeCardinality')
for j in nodes:
    if j == root_node:
        continue
    inflow = gp.quicksum((f_vars[i, j] for i in nodes if (i, j) in f_vars))
    outflow = gp.quicksum((f_vars[j, k] for k in nodes if (j, k) in f_vars))
    m.addConstr(inflow - outflow == 1, name=f'FlowConservation_{j}')
inflow_root = gp.quicksum((f_vars[i, root_node] for i in nodes if (i, root_node) in f_vars))
outflow_root = gp.quicksum((f_vars[root_node, k] for k in nodes if (root_node, k) in f_vars))
m.addConstr(outflow_root - inflow_root == n_nodes - 1, name='FlowConservation_Root')
for (i, j) in arcs:
    e = (i, j) if (i, j) in y_vars else (j, i)
    e = (i, j) if i <= j else (j, i)
    if e not in y_vars:
        raise ValueError(f'Arc ({i},{j}) does not correspond to any undirected edge.')
    m.addConstr(f_vars[i, j] <= (n_nodes - 1) * y_vars[e], name=f'FlowLink_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total construction cost: {m.objVal:.2f}')
    print('\n--- Selected Edges (y_e = 1) ---')
    for e in edges:
        if y_vars[e].X > 0.5:
            print(f'  Edge: {e[0]} -- {e[1]} (Cost: {edge_cost[e]})')
else:
    print(f'No optimal solution found. Status: {m.status}')