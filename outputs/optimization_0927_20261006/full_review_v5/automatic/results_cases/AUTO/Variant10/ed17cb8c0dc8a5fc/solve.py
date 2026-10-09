import gurobipy as gp
import pandas as pd
import numpy as np
import re
nodes_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant10/inputs/network_nodes.csv', dtype=str, keep_default_na=False)
edges_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant10/inputs/network_edges.csv', dtype=str, keep_default_na=False)
params_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant10/inputs/network_parameters.csv', dtype=str, keep_default_na=False)
nodes = nodes_df['Node'].str.strip().tolist()
n_nodes = len(nodes)
node_set = set(nodes)
edges_df['Node1'] = edges_df['Node1'].str.strip()
edges_df['Node2'] = edges_df['Node2'].str.strip()
edges_df['ConstructionCost'] = edges_df['ConstructionCost'].astype(int)

def canonical_edge(row):
    (n1, n2) = (row['Node1'], row['Node2'])
    return tuple(sorted((n1, n2)))
edges_df['EdgeKey'] = edges_df.apply(canonical_edge, axis=1)
edges = edges_df['EdgeKey'].tolist()
edge_costs = dict(zip(edges_df['EdgeKey'], edges_df['ConstructionCost']))
arcs = []
arc_to_edge = dict()
for (i, j) in edges:
    arcs.append((i, j))
    arcs.append((j, i))
    arc_to_edge[i, j] = (i, j) if (i, j) in edge_costs else (j, i)
    arc_to_edge[j, i] = (i, j) if (i, j) in edge_costs else (j, i)
root_row = params_df[params_df['Parameter'].str.strip().str.casefold() == 'rootnode']
if root_row.empty:
    raise ValueError('RootNode parameter not found in network_parameters.csv')
root_node = root_row.iloc[0]['Value'].strip()
if root_node not in node_set:
    raise ValueError(f"Root node '{root_node}' not found in node list.")
m = gp.Model('MinimumSpanningTree_FlowConnectivity')
y_vars = m.addVars(edges, vtype=gp.GRB.BINARY, name='')
f_vars = m.addVars(arcs, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((edge_costs[e] * y_vars[e] for e in edges)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((y_vars[e] for e in edges)) == n_nodes - 1, name='SpanningTreeCardinality')
for k in nodes:
    in_arcs = [(i, k) for (i, j) in arcs if j == k]
    out_arcs = [(k, j) for (i, j) in arcs if i == k]
    if k == root_node:
        m.addConstr(gp.quicksum((f_vars[a] for a in out_arcs)) - gp.quicksum((f_vars[a] for a in in_arcs)) == n_nodes - 1, name=f'FlowConservation_{k}')
    else:
        m.addConstr(gp.quicksum((f_vars[a] for a in in_arcs)) - gp.quicksum((f_vars[a] for a in out_arcs)) == 1, name=f'FlowConservation_{k}')
for (i, j) in arcs:
    e = tuple(sorted((i, j)))
    m.addConstr(f_vars[i, j] <= (n_nodes - 1) * y_vars[e], name=f'FlowEdgeLink_{i}_{j}')
m.optimize()