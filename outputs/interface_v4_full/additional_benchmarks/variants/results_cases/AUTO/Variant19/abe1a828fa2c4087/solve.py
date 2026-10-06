import gurobipy as gp
import pandas as pd
import numpy as np
nodes_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant19/inputs/network_nodes.csv', sep=',')
edges_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant19/inputs/network_edges.csv', sep=',')
params_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant19/inputs/network_parameters.csv', sep=',')

def norm_node(x):
    return str(x).strip()
nodes = nodes_df['Node'].apply(norm_node).tolist()
n = len(nodes)
if n < 2:
    raise ValueError('At least two nodes are required for a spanning tree.')
edges = []
edge_cost = {}
for idx, row in edges_df.iterrows():
    i = norm_node(row['Node1'])
    j = norm_node(row['Node2'])
    cost = float(row['ConstructionCost'])
    if i == j:
        raise ValueError(f'Self-loop detected in edge ({i},{j})')
    i_, j_ = (min(i, j), max(i, j))
    edges.append((i_, j_))
    edge_cost[i_, j_] = cost
edges = list(set(edges))
arcs = []
for i, j in edges:
    arcs.append((i, j))
    arcs.append((j, i))
root_row = params_df[params_df['Parameter'].str.strip().str.casefold() == 'rootnode']
if root_row.empty:
    raise ValueError('RootNode parameter not found in network_parameters.csv')
root = norm_node(root_row.iloc[0]['Value'])
if root not in nodes:
    raise ValueError(f"Root node '{root}' not found in node list.")
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
for i, j in arcs:
    undirected = (min(i, j), max(i, j))
    m.addConstr(f[i, j] <= M * y[undirected], name=f'FlowLink_{i}_{j}')
m.optimize()