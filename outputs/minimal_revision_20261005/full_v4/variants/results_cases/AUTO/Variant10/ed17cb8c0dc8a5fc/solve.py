import gurobipy as gp
import pandas as pd
import numpy as np
import re

def norm_id(x):
    return str(x).strip()
nodes_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant10/inputs/network_nodes.csv', sep=',')
edges_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant10/inputs/network_edges.csv', sep=',')
params_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant10/inputs/network_parameters.csv', sep=',')
nodes_df['Node'] = nodes_df['Node'].apply(norm_id)
edges_df['Node1'] = edges_df['Node1'].apply(norm_id)
edges_df['Node2'] = edges_df['Node2'].apply(norm_id)
params_df['Parameter'] = params_df['Parameter'].apply(norm_id)
params_df['Value'] = params_df['Value'].apply(norm_id)
nodes = list(nodes_df['Node'])
n = len(nodes)
node_set = set(nodes)
edges = []
edge_cost = {}
for (idx, row) in edges_df.iterrows():
    i = row['Node1']
    j = row['Node2']
    cost = row['ConstructionCost']
    if i not in node_set or j not in node_set:
        raise ValueError(f'Edge ({i},{j}) includes node not in node set.')
    key = tuple(sorted((i, j)))
    edges.append(key)
    edge_cost[key] = cost
root_row = params_df[params_df['Parameter'].str.casefold() == 'rootnode']
if root_row.empty:
    raise ValueError('RootNode parameter not found in network_parameters.csv')
root = root_row.iloc[0]['Value']
if root not in node_set:
    raise ValueError(f"Root node '{root}' not found in node set.")
arcs = []
arc_to_edge = {}
for (i, j) in edges:
    arcs.append((i, j))
    arcs.append((j, i))
    arc_to_edge[i, j] = tuple(sorted((i, j)))
    arc_to_edge[j, i] = tuple(sorted((i, j)))
m = gp.Model('MST_FlowConnectivity')
y = m.addVars(edges, vtype=gp.GRB.BINARY, name='')
f = m.addVars(arcs, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((edge_cost[e] * y[e] for e in edges)), gp.GRB.MINIMIZE)
m.addConstr(y.sum() == n - 1, name='cardinality')
for k in nodes:
    in_arcs = [(i, k) for (i, k_) in arcs if k_ == k]
    out_arcs = [(k, j) for (k_, j) in arcs if k_ == k]
    if k == root:
        m.addConstr(gp.quicksum((f[a] for a in out_arcs)) - gp.quicksum((f[a] for a in in_arcs)) == n - 1, name=f'flow_root_{k}')
    else:
        m.addConstr(gp.quicksum((f[a] for a in in_arcs)) - gp.quicksum((f[a] for a in out_arcs)) == 1, name=f'flow_node_{k}')
for arc in arcs:
    e = arc_to_edge[arc]
    m.addConstr(f[arc] <= (n - 1) * y[e], name=f'flow_link_{arc[0]}_{arc[1]}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for e in edges:
        print(f'y[{e[0]},{e[1]}] {y[e].VarName} {y[e].X}')
    for arc in arcs:
        print(f'f[{arc[0]},{arc[1]}] {f[arc].VarName} {f[arc].X}')
else:
    print(f'Solver status: {m.status}')