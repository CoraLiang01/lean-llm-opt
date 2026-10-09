import gurobipy as gp
import pandas as pd
import numpy as np
import re
nodes_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant10/inputs/network_nodes.csv', dtype=str, keep_default_na=False)
edges_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant10/inputs/network_edges.csv', dtype=str, keep_default_na=False)
params_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant10/inputs/network_parameters.csv', dtype=str, keep_default_na=False)
nodes = nodes_df['Node'].str.strip().tolist()
n_nodes = len(nodes)
nodes_set = set(nodes)
edges = []
edge_cost = {}
for (idx, row) in edges_df.iterrows():
    u = row['Node1'].strip()
    v = row['Node2'].strip()
    if u not in nodes_set or v not in nodes_set:
        raise ValueError(f'Edge ({u},{v}) references unknown node(s).')
    edge = tuple(sorted([u, v]))
    if edge in edge_cost:
        raise ValueError(f'Duplicate edge between {u} and {v}.')
    edges.append(edge)
    try:
        cost = int(row['ConstructionCost'])
    except Exception:
        raise ValueError(f"Invalid ConstructionCost for edge ({u},{v}): {row['ConstructionCost']}")
    edge_cost[edge] = cost
arcs = []
arc_to_edge = {}
for (u, v) in edges:
    arcs.append((u, v))
    arcs.append((v, u))
    arc_to_edge[u, v] = (u, v) if (u, v) in edge_cost else (v, u)
    arc_to_edge[v, u] = (u, v) if (u, v) in edge_cost else (v, u)
root_row = params_df[params_df['Parameter'].str.strip().str.casefold() == 'rootnode']
if root_row.empty:
    raise ValueError('RootNode parameter not found in network_parameters.csv')
root_node = root_row.iloc[0]['Value'].strip()
if root_node not in nodes_set:
    raise ValueError(f"Root node '{root_node}' not found in node list.")
m = gp.Model('MinimumSpanningTree_FlowBased')
y_vars = m.addVars(edges, vtype=gp.GRB.BINARY, name='')
f_vars = m.addVars(arcs, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((edge_cost[e] * y_vars[e] for e in edges)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((y_vars[e] for e in edges)) == n_nodes - 1, name='spanning_tree_cardinality')
for k in nodes:
    if k == root_node:
        continue
    inflow = gp.quicksum((f_vars[i, k] for (i, j) in arcs if j == k))
    outflow = gp.quicksum((f_vars[k, j] for (i, j) in arcs if i == k))
    m.addConstr(inflow - outflow == 1, name=f'flow_conservation_{k}')
inflow_root = gp.quicksum((f_vars[i, root_node] for (i, j) in arcs if j == root_node))
outflow_root = gp.quicksum((f_vars[root_node, j] for (i, j) in arcs if i == root_node))
m.addConstr(outflow_root - inflow_root == n_nodes - 1, name='flow_conservation_root')
for (i, j) in arcs:
    e = tuple(sorted([i, j]))
    m.addConstr(f_vars[i, j] <= (n_nodes - 1) * y_vars[e], name=f'flow_link_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total construction cost: {m.objVal:.0f}')
    print('\n--- Selected Links (edges) ---')
    for e in edges:
        if y_vars[e].X > 0.5:
            print(f'  Link: {e[0]} -- {e[1]} (Cost: {edge_cost[e]})')
    print('\n--- Flow on Arcs (nonzero) ---')
    for (i, j) in arcs:
        val = f_vars[i, j].X
        if val > 1e-06:
            print(f'  Flow from {i} to {j}: {val:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')