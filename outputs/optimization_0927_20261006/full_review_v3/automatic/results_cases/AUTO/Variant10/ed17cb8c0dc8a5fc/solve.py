import gurobipy as gp
import pandas as pd
import numpy as np
import re
nodes_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant10/inputs/network_nodes.csv', sep=',', dtype=str, keep_default_na=False)
edges_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant10/inputs/network_edges.csv', sep=',', dtype=str, keep_default_na=False)
params_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant10/inputs/network_parameters.csv', sep=',', dtype=str, keep_default_na=False)
nodes = nodes_df['Node'].str.strip().tolist()
n_nodes = len(nodes)
node_set = set(nodes)
edges_df['Node1'] = edges_df['Node1'].str.strip()
edges_df['Node2'] = edges_df['Node2'].str.strip()
edges_df['ConstructionCost'] = edges_df['ConstructionCost'].astype(float)
edges = []
edge_cost = {}
for (idx, row) in edges_df.iterrows():
    n1 = row['Node1']
    n2 = row['Node2']
    if n1 not in node_set or n2 not in node_set:
        raise ValueError(f'Edge ({n1}, {n2}) contains node not in node set.')
    e = frozenset([n1, n2])
    if n1 == n2:
        raise ValueError(f'Self-loop edge ({n1}, {n2}) is not allowed.')
    if e in edge_cost:
        raise ValueError(f'Duplicate undirected edge between {n1} and {n2}.')
    edges.append(e)
    edge_cost[e] = row['ConstructionCost']
arcs = []
arc_to_edge = {}
for e in edges:
    (n1, n2) = tuple(e)
    arcs.append((n1, n2))
    arcs.append((n2, n1))
    arc_to_edge[n1, n2] = e
    arc_to_edge[n2, n1] = e
root_row = params_df.loc[params_df['Parameter'].str.strip().str.casefold() == 'rootnode']
if root_row.empty:
    raise ValueError('RootNode parameter not found in network_parameters.csv')
root_node = root_row.iloc[0]['Value'].strip()
if root_node not in node_set:
    raise ValueError(f"Root node '{root_node}' not found in node set.")
m = gp.Model('MinimumSpanningTree_FlowBased')
y_vars = m.addVars(edges, vtype=gp.GRB.BINARY, name='')
f_vars = m.addVars(arcs, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((edge_cost[e] * y_vars[e] for e in edges)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((y_vars[e] for e in edges)) == n_nodes - 1, name='SpanningTreeSize')
for v in nodes:
    inflow = gp.quicksum((f_vars[i, v] for i in nodes if (i, v) in f_vars))
    outflow = gp.quicksum((f_vars[v, j] for j in nodes if (v, j) in f_vars))
    if v == root_node:
        m.addConstr(outflow - inflow == n_nodes - 1, name=f'FlowConservation_{v}')
    else:
        m.addConstr(inflow - outflow == 1, name=f'FlowConservation_{v}')
for (i, j) in arcs:
    e = arc_to_edge[i, j]
    m.addConstr(f_vars[i, j] <= (n_nodes - 1) * y_vars[e], name=f'FlowLink_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total construction cost: {m.objVal:.2f}')
    print('\n--- Selected Links (edges) ---')
    for e in edges:
        if y_vars[e].X > 0.5:
            (n1, n2) = tuple(e)
            print(f'  Link: {n1} -- {n2} (Cost: {edge_cost[e]:.2f})')
    print('\n--- Nonzero Flows (f_ij) ---')
    for (i, j) in arcs:
        val = f_vars[i, j].X
        if val > 1e-06:
            print(f'  Flow from {i} to {j}: {val:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')