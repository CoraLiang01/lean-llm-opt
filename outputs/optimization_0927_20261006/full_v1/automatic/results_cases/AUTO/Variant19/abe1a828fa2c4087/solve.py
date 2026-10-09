import gurobipy as gp
import pandas as pd
import numpy as np
import re

def find_col(df, pattern):
    for col in df.columns:
        if re.search(pattern, col, re.IGNORECASE):
            return col
    raise KeyError(f"Could not find a column matching pattern '{pattern}'")
nodes_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant19/inputs/network_nodes.csv', dtype=str, keep_default_na=False)
edges_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant19/inputs/network_edges.csv', dtype=str, keep_default_na=False)
edges_df['ConstructionCost'] = edges_df['ConstructionCost'].astype(float)
params_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant19/inputs/network_parameters.csv', dtype=str, keep_default_na=False)
node_list = nodes_df['Node'].apply(lambda x: x.strip())
N = list(node_list)
n = len(N)
edges = []
edge_cost = {}
for (idx, row) in edges_df.iterrows():
    i = row['Node1'].strip()
    j = row['Node2'].strip()
    if i < j:
        edge = (i, j)
    else:
        edge = (j, i)
    edges.append(edge)
    edge_cost[edge] = row['ConstructionCost']
edges = list(dict.fromkeys(edges))
arcs = []
for (i, j) in edges:
    arcs.append((i, j))
    arcs.append((j, i))
root_row = params_df.loc[params_df['Parameter'].str.strip().str.casefold() == 'rootnode']
if root_row.empty:
    raise ValueError('RootNode parameter not found in network_parameters.csv')
root = root_row.iloc[0]['Value'].strip()
if root not in N:
    raise ValueError(f"Root node '{root}' not found in node list.")
M = n - 1
m = gp.Model('MinCostSpanningTree_SCF')
y_vars = m.addVars(edges, vtype=gp.GRB.BINARY, name='')
f_vars = m.addVars(arcs, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((edge_cost[edge] * y_vars[edge] for edge in edges)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((y_vars[edge] for edge in edges)) == n - 1, name='EdgeCount')
m.addConstr(gp.quicksum((f_vars[root, j] for j in N if (root, j) in f_vars)) == n - 1, name='RootFlowBalance')
for k in N:
    if k == root:
        continue
    m.addConstr(gp.quicksum((f_vars[i, k] for i in N if (i, k) in f_vars)) - gp.quicksum((f_vars[k, j] for j in N if (k, j) in f_vars)) == 1, name=f'FlowBalance_{k}')
for (i, j) in arcs:
    undirected = (i, j) if i < j else (j, i)
    m.addConstr(f_vars[i, j] <= M * y_vars[undirected], name=f'FlowToEdge_{i}_{j}')
m.optimize()