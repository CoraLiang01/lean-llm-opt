import gurobipy as gp
import pandas as pd
import numpy as np
import re

def norm_str(x):
    return str(x).strip().casefold()
nodes_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant10/inputs/network_nodes.csv', dtype=str, keep_default_na=False)
edges_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant10/inputs/network_edges.csv', dtype=str, keep_default_na=False)
params_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant10/inputs/network_parameters.csv', dtype=str, keep_default_na=False)
nodes_df['Node_norm'] = nodes_df['Node'].apply(norm_str)
nodes = list(nodes_df['Node'].values)
nodes_norm = list(nodes_df['Node_norm'].values)
n_nodes = len(nodes)
edges_df['Node1_norm'] = edges_df['Node1'].apply(norm_str)
edges_df['Node2_norm'] = edges_df['Node2'].apply(norm_str)
edges_df['ConstructionCost'] = edges_df['ConstructionCost'].astype(float)

def undirected_edge_key(row):
    (u, v) = (row['Node1_norm'], row['Node2_norm'])
    return tuple(sorted([u, v]))
edges_df['edge_key'] = edges_df.apply(undirected_edge_key, axis=1)
edge_keys = list(edges_df['edge_key'].values)
edge_key_to_nodes = {ek: (edges_df.loc[i, 'Node1'], edges_df.loc[i, 'Node2']) for (i, ek) in enumerate(edge_keys)}
edge_cost = dict(zip(edge_keys, edges_df['ConstructionCost']))
arcs = []
arc_to_edge_key = dict()
for ek in edge_keys:
    (u, v) = ek
    arcs.append((u, v))
    arcs.append((v, u))
    arc_to_edge_key[u, v] = ek
    arc_to_edge_key[v, u] = ek
params_df['Parameter_norm'] = params_df['Parameter'].apply(norm_str)
root_row = params_df[params_df['Parameter_norm'] == 'rootnode']
if root_row.empty:
    raise ValueError('RootNode parameter not found in network_parameters.csv')
root_node = root_row.iloc[0]['Value']
root_node_norm = norm_str(root_node)
if root_node_norm not in nodes_norm:
    raise ValueError(f"Root node '{root_node}' not found in node list.")
norm_to_node = dict(zip(nodes_norm, nodes))
m = gp.Model('MinimumSpanningTree_FlowConnectivity')
y_vars = m.addVars(edge_keys, vtype=gp.GRB.BINARY, name='')
f_vars = m.addVars(arcs, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((edge_cost[ek] * y_vars[ek] for ek in edge_keys)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((y_vars[ek] for ek in edge_keys)) == n_nodes - 1, name='spanning_tree_cardinality')
for v_norm in nodes_norm:
    inflow = gp.quicksum((f_vars[u, v_norm] for u in nodes_norm if (u, v_norm) in f_vars))
    outflow = gp.quicksum((f_vars[v_norm, w] for w in nodes_norm if (v_norm, w) in f_vars))
    if v_norm == root_node_norm:
        m.addConstr(outflow - inflow == n_nodes - 1, name=f'flow_root_{norm_to_node[v_norm]}')
    else:
        m.addConstr(inflow - outflow == 1, name=f'flow_node_{norm_to_node[v_norm]}')
for arc in arcs:
    ek = arc_to_edge_key[arc]
    m.addConstr(f_vars[arc] <= (n_nodes - 1) * y_vars[ek], name=f'flow_link_{arc[0]}_{arc[1]}')
m.optimize()