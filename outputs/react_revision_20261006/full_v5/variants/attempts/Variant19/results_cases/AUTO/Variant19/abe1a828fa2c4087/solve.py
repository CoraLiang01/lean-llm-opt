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
if n < 2:
    raise ValueError('At least two nodes are required for a spanning tree.')
root_row = params_df[params_df['Parameter'].str.strip().str.casefold() == 'rootnode']
if root_row.empty:
    raise ValueError('RootNode parameter not found in network_parameters.csv')
root = root_row.iloc[0]['Value'].strip()
if root not in nodes:
    raise ValueError(f"Root node '{root}' not found in node list.")
edges_df['Node1'] = edges_df['Node1'].astype(str).str.strip()
edges_df['Node2'] = edges_df['Node2'].astype(str).str.strip()
edges_df['ConstructionCost'] = edges_df['ConstructionCost'].astype(float)
E = []
edge_cost = {}
for (idx, row) in edges_df.iterrows():
    (i, j) = (row['Node1'], row['Node2'])
    if i == j:
        raise ValueError(f'Self-loop detected in edge ({i}, {j})')
    key = tuple(sorted((i, j)))
    if key in edge_cost:
        raise ValueError(f'Duplicate undirected edge ({i}, {j}) in input.')
    E.append(key)
    edge_cost[key] = row['ConstructionCost']
for (i, j) in E:
    if i not in nodes or j not in nodes:
        raise ValueError(f'Edge ({i}, {j}) contains node(s) not in node list.')
A = []
arc_to_edge = {}
for (i, j) in E:
    A.append((i, j))
    A.append((j, i))
    arc_to_edge[i, j] = tuple(sorted((i, j)))
    arc_to_edge[j, i] = tuple(sorted((i, j)))
M = n - 1
m = gp.Model('MST_SingleCommodityFlow')
y = m.addVars(E, vtype=gp.GRB.BINARY, name='')
f = m.addVars(A, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((edge_cost[e] * y[e] for e in E)), gp.GRB.MINIMIZE)
m.addConstr(y.sum() == n - 1, name='EdgeCount')
m.addConstr(gp.quicksum((f[root, j] for j in nodes if (root, j) in A)) == n - 1, name='RootFlow')
for k in nodes:
    if k == root:
        continue
    m.addConstr(gp.quicksum((f[i, k] for i in nodes if (i, k) in A)) - gp.quicksum((f[k, j] for j in nodes if (k, j) in A)) == 1, name=f'FlowBalance_{k}')
for (i, j) in A:
    undirected = arc_to_edge[i, j]
    m.addConstr(f[i, j] <= M * y[undirected], name=f'FlowLink_{i}_{j}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for e in E:
        print(f'y[{e[0]},{e[1]}] {y[e].VarName} {y[e].X}')
    for a in A:
        print(f'f[{a[0]},{a[1]}] {f[a].VarName} {f[a].X}')
else:
    print(f'Solver status: {m.status}')