import gurobipy as gp
import pandas as pd
import numpy as np
import re

def find_col(df, pattern):
    for col in df.columns:
        if re.search(pattern, col, re.IGNORECASE):
            return col
    raise KeyError(f"Could not find a column matching pattern '{pattern}'")
nodes_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant19/inputs/network_nodes.csv', sep=',')
edges_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant19/inputs/network_edges.csv', sep=',')
params_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant19/inputs/network_parameters.csv', sep=',')
nodes_df['Node'] = nodes_df['Node'].astype(str).str.strip()
nodes = list(nodes_df['Node'])
n = len(nodes)
edges_df['Node1'] = edges_df['Node1'].astype(str).str.strip()
edges_df['Node2'] = edges_df['Node2'].astype(str).str.strip()
edges = []
edge_cost = {}
for (idx, row) in edges_df.iterrows():
    (i, j) = (row['Node1'], row['Node2'])
    if i not in nodes or j not in nodes:
        raise ValueError(f'Edge ({i},{j}) contains node not in node set.')
    key = tuple(sorted([i, j]))
    if key in edge_cost:
        raise ValueError(f'Duplicate undirected edge: {key}')
    edges.append(key)
    edge_cost[key] = float(row['ConstructionCost'])
arcs = []
arc_to_edge = {}
for (i, j) in edges:
    arcs.append((i, j))
    arcs.append((j, i))
    arc_to_edge[i, j] = tuple(sorted([i, j]))
    arc_to_edge[j, i] = tuple(sorted([i, j]))
params_df['Parameter'] = params_df['Parameter'].astype(str).str.strip().str.casefold()
params_df['Value'] = params_df['Value'].astype(str).str.strip()
root_row = params_df[params_df['Parameter'] == 'rootnode']
if root_row.empty:
    raise ValueError('RootNode parameter not found in network_parameters.csv')
root = root_row.iloc[0]['Value']
if root not in nodes:
    raise ValueError(f"Root node '{root}' not found in node set.")
M = n - 1
m = gp.Model('MST_SingleCommodityFlow')
y = m.addVars(edges, vtype=gp.GRB.BINARY, name='')
f = m.addVars(arcs, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((edge_cost[e] * y[e] for e in edges)), gp.GRB.MINIMIZE)
m.addConstr(y.sum() == n - 1, name='edge_count')
out_arcs_root = [(root, j) for j in nodes if j != root and (root, j) in arcs]
in_arcs_root = [(j, root) for j in nodes if j != root and (j, root) in arcs]
m.addConstr(gp.quicksum((f[a] for a in out_arcs_root)) - gp.quicksum((f[a] for a in in_arcs_root)) == n - 1, name='root_flow_balance')
for k in nodes:
    if k == root:
        continue
    in_arcs = [(j, k) for j in nodes if j != k and (j, k) in arcs]
    out_arcs = [(k, j) for j in nodes if j != k and (k, j) in arcs]
    m.addConstr(gp.quicksum((f[a] for a in in_arcs)) - gp.quicksum((f[a] for a in out_arcs)) == 1, name=f'flow_balance_{k}')
for (i, j) in arcs:
    undirected = arc_to_edge[i, j]
    m.addConstr(f[i, j] <= M * y[undirected], name=f'flow_link_{i}_{j}')
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')