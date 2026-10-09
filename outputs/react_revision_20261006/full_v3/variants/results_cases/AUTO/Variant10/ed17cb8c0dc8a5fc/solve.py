import gurobipy as gp
import pandas as pd
import numpy as np
import re

def find_col(df, pattern):
    for col in df.columns:
        if re.search(pattern, col, re.IGNORECASE):
            return col
    raise KeyError(f"Could not find a column matching pattern '{pattern}'")
nodes_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant10/inputs/network_nodes.csv', sep=',')
edges_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant10/inputs/network_edges.csv', sep=',')
params_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant10/inputs/network_parameters.csv', sep=',')
nodes_df['Node'] = nodes_df['Node'].astype(str).str.strip()
node_list = nodes_df['Node'].tolist()
n_nodes = len(node_list)
node_set = set(node_list)
if len(node_set) != n_nodes:
    raise ValueError('Duplicate node identifiers found in network_nodes.csv.')
root_row = params_df[params_df['Parameter'].str.strip().str.casefold() == 'rootnode']
if root_row.empty:
    raise ValueError('RootNode parameter not found in network_parameters.csv.')
root_node = root_row.iloc[0]['Value'].strip()
if root_node not in node_set:
    raise ValueError(f"Root node '{root_node}' not found in node list.")
edges_df['Node1'] = edges_df['Node1'].astype(str).str.strip()
edges_df['Node2'] = edges_df['Node2'].astype(str).str.strip()
edges_df['ConstructionCost'] = edges_df['ConstructionCost'].astype(float)
if not set(edges_df['Node1']).issubset(node_set) or not set(edges_df['Node2']).issubset(node_set):
    missing = (set(edges_df['Node1']) | set(edges_df['Node2'])) - node_set
    raise ValueError(f'Edge endpoints not found in node list: {missing}')
edges = []
edge_cost = {}
for (idx, row) in edges_df.iterrows():
    i = row['Node1']
    j = row['Node2']
    if i == j:
        raise ValueError(f'Self-loop edge found: {i}-{j}')
    key = tuple(sorted((i, j)))
    if key in edge_cost:
        raise ValueError(f'Duplicate undirected edge found: {key}')
    edges.append(key)
    edge_cost[key] = row['ConstructionCost']
arc_list = []
arc_to_edge = {}
for (i, j) in edges:
    arc_list.append((i, j))
    arc_list.append((j, i))
    arc_to_edge[i, j] = (i, j) if (i, j) in edge_cost else (j, i)
    arc_to_edge[j, i] = (i, j) if (i, j) in edge_cost else (j, i)

def solve_mst_flow():
    m = gp.Model('MST_FlowConnectivity')
    m.setParam('MIPGap', 0.0001)
    y = m.addVars(edges, vtype=gp.GRB.BINARY, name='')
    f = m.addVars(arc_list, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((edge_cost[e] * y[e] for e in edges)), gp.GRB.MINIMIZE)
    m.addConstr(y.sum() == n_nodes - 1, name='TreeSize')
    for k in node_list:
        if k == root_node:
            m.addConstr(gp.quicksum((f[root_node, j] for j in node_list if (root_node, j) in f)) - gp.quicksum((f[j, root_node] for j in node_list if (j, root_node) in f)) == n_nodes - 1, name=f'FlowConsv_root')
        else:
            m.addConstr(gp.quicksum((f[i, k] for i in node_list if (i, k) in f)) - gp.quicksum((f[k, j] for j in node_list if (k, j) in f)) == 1, name=f'FlowConsv_{k}')
    for (i, j) in arc_list:
        undirected = tuple(sorted((i, j)))
        m.addConstr(f[i, j] <= (n_nodes - 1) * y[undirected], name=f'FlowLink_{i}_{j}')
    m.optimize()
    return m
m = solve_mst_flow()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')