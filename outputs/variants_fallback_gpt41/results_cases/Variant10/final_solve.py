import gurobipy as gp
import pandas as pd
import numpy as np
import re
nodes_df = pd.read_csv('benchmark_dataset/Variant10/inputs/network_nodes.csv', dtype=str, keep_default_na=False)
edges_df = pd.read_csv('benchmark_dataset/Variant10/inputs/network_edges.csv', dtype=str, keep_default_na=False)
params_df = pd.read_csv('benchmark_dataset/Variant10/inputs/network_parameters.csv', dtype=str, keep_default_na=False)
nodes_df['Node'] = nodes_df['Node'].str.strip()
nodes = list(nodes_df['Node'])
n_nodes = len(nodes)
if n_nodes < 2:
    raise ValueError('At least two nodes are required for a spanning tree.')
edges_df['Node1'] = edges_df['Node1'].str.strip()
edges_df['Node2'] = edges_df['Node2'].str.strip()
edges_df['ConstructionCost'] = edges_df['ConstructionCost'].astype(float)
for (idx, row) in edges_df.iterrows():
    if row['Node1'] not in nodes or row['Node2'] not in nodes:
        raise ValueError(f"Edge endpoints {row['Node1']}, {row['Node2']} not in node set.")

def undirected_edge_key(u, v):
    return tuple(sorted((u, v)))
edges = []
edge_cost = {}
for (idx, row) in edges_df.iterrows():
    u = row['Node1']
    v = row['Node2']
    key = undirected_edge_key(u, v)
    if key in edge_cost:
        raise ValueError(f'Duplicate undirected edge between {u} and {v}.')
    edges.append(key)
    edge_cost[key] = row['ConstructionCost']
params_df['Parameter'] = params_df['Parameter'].str.strip().str.casefold()
root_row = params_df[params_df['Parameter'] == 'rootnode']
if root_row.empty:
    raise ValueError('RootNode parameter not found in network_parameters.csv.')
root_node = root_row.iloc[0]['Value'].strip()
if root_node not in nodes:
    raise ValueError(f"Root node '{root_node}' not found in node list.")
arcs = []
arc_to_edge = {}
for (u, v) in edges:
    arcs.append((u, v))
    arcs.append((v, u))
    arc_to_edge[u, v] = undirected_edge_key(u, v)
    arc_to_edge[v, u] = undirected_edge_key(u, v)
m = gp.Model('MinimumSpanningTree_FlowBased')
y_vars = m.addVars(edges, vtype=gp.GRB.BINARY, name='')
f_vars = m.addVars(arcs, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((edge_cost[e] * y_vars[e] for e in edges)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((y_vars[e] for e in edges)) == n_nodes - 1, name='spanning_tree_cardinality')
for node in nodes:
    inflow = gp.quicksum((f_vars[i, node] for i in nodes if (i, node) in f_vars))
    outflow = gp.quicksum((f_vars[node, j] for j in nodes if (node, j) in f_vars))
    if node == root_node:
        m.addConstr(outflow - inflow == n_nodes - 1, name=f'flow_root_{node}')
    else:
        m.addConstr(inflow - outflow == 1, name=f'flow_node_{node}')
for arc in arcs:
    e = arc_to_edge[arc]
    m.addConstr(f_vars[arc] <= (n_nodes - 1) * y_vars[e], name=f'flow_link_{arc[0]}_{arc[1]}')
m.optimize()