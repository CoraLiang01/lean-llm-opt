import gurobipy as gp
import pandas as pd
import numpy as np
import re

def find_col(df, pattern):
    for col in df.columns:
        if re.search(pattern, col, re.IGNORECASE):
            return col
    raise KeyError(f"Could not find a column matching pattern '{pattern}'")
nodes_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant10/inputs/network_nodes.csv', dtype=str, keep_default_na=False)
edges_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant10/inputs/network_edges.csv', dtype=str, keep_default_na=False)
params_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant10/inputs/network_parameters.csv', dtype=str, keep_default_na=False)
nodes = nodes_df['Node'].str.strip().tolist()
n_nodes = len(nodes)
node_set = set(nodes)
edges_df['Node1'] = edges_df['Node1'].str.strip()
edges_df['Node2'] = edges_df['Node2'].str.strip()
edges = []
edge_cost = {}
for (idx, row) in edges_df.iterrows():
    i = row['Node1']
    j = row['Node2']
    edge = tuple(sorted((i, j)))
    edges.append(edge)
    try:
        cost = int(row['ConstructionCost'])
    except Exception:
        raise ValueError(f"Invalid ConstructionCost for edge {edge}: {row['ConstructionCost']}")
    edge_cost[edge] = cost
arcs = []
arc_to_edge = {}
for (i, j) in edges:
    arcs.append((i, j))
    arcs.append((j, i))
    arc_to_edge[i, j] = tuple(sorted((i, j)))
    arc_to_edge[j, i] = tuple(sorted((i, j)))
root_row = params_df.loc[params_df['Parameter'].str.strip().str.casefold() == 'rootnode']
if root_row.empty:
    raise ValueError('RootNode parameter not found in network_parameters.csv')
root_node = root_row.iloc[0]['Value'].strip()
if root_node not in node_set:
    raise ValueError(f"Root node '{root_node}' not found in node list.")
m = gp.Model('MinimumSpanningTree_FlowBased')
y_vars = m.addVars(edges, vtype=gp.GRB.BINARY, name='')
f_vars = m.addVars(arcs, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((edge_cost[edge] * y_vars[edge] for edge in edges)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((y_vars[edge] for edge in edges)) == n_nodes - 1, name='SpanningTreeCardinality')
for k in nodes:
    in_arcs = [(i, k) for i in nodes if (i, k) in arcs]
    out_arcs = [(k, j) for j in nodes if (k, j) in arcs]
    if k == root_node:
        m.addConstr(gp.quicksum((f_vars[arc] for arc in out_arcs)) - gp.quicksum((f_vars[arc] for arc in in_arcs)) == n_nodes - 1, name=f'FlowConservation_root_{k}')
    else:
        m.addConstr(gp.quicksum((f_vars[arc] for arc in in_arcs)) - gp.quicksum((f_vars[arc] for arc in out_arcs)) == 1, name=f'FlowConservation_{k}')
for arc in arcs:
    edge = arc_to_edge[arc]
    m.addConstr(f_vars[arc] <= (n_nodes - 1) * y_vars[edge], name=f'FlowLink_{arc[0]}_{arc[1]}')
m.optimize()