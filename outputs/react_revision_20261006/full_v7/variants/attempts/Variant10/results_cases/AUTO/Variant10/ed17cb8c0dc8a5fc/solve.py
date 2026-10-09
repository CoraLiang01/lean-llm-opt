import gurobipy as gp
import pandas as pd
import numpy as np
import re

def normalize_id(x):
    return str(x).strip()
nodes_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant10/inputs/network_nodes.csv', dtype=str, keep_default_na=False)
edges_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant10/inputs/network_edges.csv', dtype=str, keep_default_na=False)
params_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant10/inputs/network_parameters.csv', dtype=str, keep_default_na=False)
nodes_df['Node'] = nodes_df['Node'].apply(normalize_id)
node_list = nodes_df['Node'].tolist()
n_nodes = len(node_list)
node_set = set(node_list)
if len(node_set) != n_nodes:
    raise ValueError('Duplicate node identifiers found in network_nodes.csv.')
edges_df['Node1'] = edges_df['Node1'].apply(normalize_id)
edges_df['Node2'] = edges_df['Node2'].apply(normalize_id)
try:
    edges_df['ConstructionCost'] = edges_df['ConstructionCost'].astype(int)
except Exception as e:
    raise ValueError('ConstructionCost column in network_edges.csv must be convertible to int.') from e
for (idx, row) in edges_df.iterrows():
    if row['Node1'] not in node_set or row['Node2'] not in node_set:
        raise ValueError(f"Edge ({row['Node1']}, {row['Node2']}) contains node(s) not in node set.")

def edge_key(i, j):
    (a, b) = sorted([i, j])
    return (a, b)
edge_tuples = []
edge_costs = {}
for (idx, row) in edges_df.iterrows():
    i = row['Node1']
    j = row['Node2']
    key = edge_key(i, j)
    if key in edge_costs:
        raise ValueError(f'Duplicate undirected edge between {i} and {j} in network_edges.csv.')
    edge_tuples.append(key)
    edge_costs[key] = row['ConstructionCost']
arc_tuples = []
arc_to_edge = {}
for (i, j) in edge_tuples:
    arc_tuples.append((i, j))
    arc_tuples.append((j, i))
    arc_to_edge[i, j] = (i, j)
    arc_to_edge[j, i] = (i, j)
params_df['Parameter'] = params_df['Parameter'].apply(lambda x: x.strip())
root_row = params_df[params_df['Parameter'].str.casefold() == 'rootnode']
if root_row.empty:
    raise ValueError('RootNode parameter not found in network_parameters.csv.')
root_node = normalize_id(root_row.iloc[0]['Value'])
if root_node not in node_set:
    raise ValueError(f"Root node '{root_node}' not found in node set.")
m = gp.Model('MST_FlowConnectivity')
y_vars = m.addVars(edge_tuples, vtype=gp.GRB.BINARY, name='')
f_vars = m.addVars(arc_tuples, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((edge_costs[e] * y_vars[e] for e in edge_tuples)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((y_vars[e] for e in edge_tuples)) == n_nodes - 1, name='spanning_cardinality')
for k in node_list:
    in_arcs = [(i, k) for i in node_list if (i, k) in arc_tuples]
    out_arcs = [(k, j) for j in node_list if (k, j) in arc_tuples]
    inflow = gp.quicksum((f_vars[a] for a in in_arcs))
    outflow = gp.quicksum((f_vars[a] for a in out_arcs))
    if k == root_node:
        m.addConstr(outflow - inflow == n_nodes - 1, name=f'flow_root')
    else:
        m.addConstr(inflow - outflow == 1, name=f'flow_{k}')
for arc in arc_tuples:
    e = arc_to_edge[arc]
    m.addConstr(f_vars[arc] <= (n_nodes - 1) * y_vars[e], name=f'flow_link_{arc[0]}_{arc[1]}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for e in edge_tuples:
        print(f'{y_vars[e].VarName} {y_vars[e].X}')
    for arc in arc_tuples:
        print(f'{f_vars[arc].VarName} {f_vars[arc].X}')
else:
    print(f'Solver status: {m.status}')