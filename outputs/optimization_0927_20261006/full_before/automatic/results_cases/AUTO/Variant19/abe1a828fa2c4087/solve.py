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
nodes = nodes_df['Node'].astype(str).str.strip().tolist()
n = len(nodes)
edges = []
edge_cost = {}
for (idx, row) in edges_df.iterrows():
    i = str(row['Node1']).strip()
    j = str(row['Node2']).strip()
    if i < j:
        e = (i, j)
    else:
        e = (j, i)
    edges.append(e)
    edge_cost[e] = int(row['ConstructionCost'])
edges = list(dict.fromkeys(edges))
arcs = []
for (i, j) in edges:
    arcs.append((i, j))
    arcs.append((j, i))
root_row = params_df.loc[params_df['Parameter'].str.strip().str.casefold() == 'rootnode']
if root_row.empty:
    raise ValueError('RootNode parameter not found in network_parameters.csv')
root = str(root_row.iloc[0]['Value']).strip()
M = n - 1
m = gp.Model('MST_SingleCommodityFlow')
y = m.addVars(edges, vtype=gp.GRB.BINARY, name='')
f = m.addVars(arcs, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((edge_cost[e] * y[e] for e in edges)), gp.GRB.MINIMIZE)
m.addConstr(y.sum() == n - 1, name='EdgeCount')
out_arcs_root = [(root, j) for j in nodes if j != root and (root, j) in arcs]
in_arcs_root = [(j, root) for j in nodes if j != root and (j, root) in arcs]
m.addConstr(gp.quicksum((f[a] for a in out_arcs_root)) - gp.quicksum((f[a] for a in in_arcs_root)) == n - 1, name='RootFlowBalance')
for k in nodes:
    if k == root:
        continue
    in_arcs = [(i, k) for i in nodes if i != k and (i, k) in arcs]
    out_arcs = [(k, j) for j in nodes if j != k and (k, j) in arcs]
    m.addConstr(gp.quicksum((f[a] for a in in_arcs)) - gp.quicksum((f[a] for a in out_arcs)) == 1, name=f'FlowBalance_{k}')
for (i, j) in arcs:
    undirected = (i, j) if i < j else (j, i)
    m.addConstr(f[i, j] <= M * y[undirected], name=f'FlowLink_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('\n--- Selected Edges (y_ij = 1) ---')
    for e in edges:
        if y[e].X > 0.5:
            print(f'  Edge {e[0]} - {e[1]} (Cost: {edge_cost[e]})')
    print('\n--- Nonzero Flows (f_ij > 0) ---')
    for a in arcs:
        if f[a].X > 1e-06:
            print(f'  Flow {a[0]} -> {a[1]}: {f[a].X:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')