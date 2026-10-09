import gurobipy as gp
import pandas as pd
import numpy as np
import re

def find_col(df, pattern):
    for col in df.columns:
        if re.search(pattern, col, re.IGNORECASE):
            return col
    raise KeyError(f"Could not find a column matching pattern '{pattern}'")

def solve_mst_single_commodity_flow():
    nodes_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant19/inputs/network_nodes.csv', sep=',')
    edges_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant19/inputs/network_edges.csv', sep=',')
    params_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant19/inputs/network_parameters.csv', sep=',')
    nodes = nodes_df['Node'].astype(str).str.strip().tolist()
    n = len(nodes)
    node_set = set(nodes)

    def norm_edge(i, j):
        return tuple(sorted((str(i).strip(), str(j).strip())))
    undirected_edges = []
    edge_cost = dict()
    for (idx, row) in edges_df.iterrows():
        i = str(row['Node1']).strip()
        j = str(row['Node2']).strip()
        cost = row['ConstructionCost']
        e = norm_edge(i, j)
        if e in edge_cost:
            raise ValueError(f'Duplicate undirected edge between {e[0]} and {e[1]}')
        undirected_edges.append(e)
        edge_cost[e] = cost
    for (i, j) in undirected_edges:
        if i not in node_set or j not in node_set:
            raise ValueError(f'Edge ({i},{j}) has endpoint not in node set')
    directed_arcs = []
    arc_to_undirected = dict()
    for (i, j) in undirected_edges:
        directed_arcs.append((i, j))
        directed_arcs.append((j, i))
        arc_to_undirected[i, j] = norm_edge(i, j)
        arc_to_undirected[j, i] = norm_edge(i, j)
    root_row = params_df[params_df['Parameter'].str.strip().str.casefold() == 'rootnode']
    if root_row.empty:
        raise ValueError('RootNode parameter not found in network_parameters.csv')
    root = str(root_row.iloc[0]['Value']).strip()
    if root not in node_set:
        raise ValueError(f"Root node '{root}' not in node set")
    M = n - 1
    m = gp.Model('MST_SingleCommodityFlow')
    m.Params.MIPGap = 0.0001
    y = m.addVars(undirected_edges, vtype=gp.GRB.BINARY, name='')
    f = m.addVars(directed_arcs, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((edge_cost[e] * y[e] for e in undirected_edges)), gp.GRB.MINIMIZE)
    m.addConstr(y.sum() == n - 1, name='EdgeCount')
    m.addConstr(gp.quicksum((f[root, j] for j in nodes if (root, j) in f and j != root)) == n - 1, name='RootFlowOut')
    for k in nodes:
        if k == root:
            continue
        inflow = gp.quicksum((f[i, k] for i in nodes if (i, k) in f and i != k))
        outflow = gp.quicksum((f[k, j] for j in nodes if (k, j) in f and j != k))
        m.addConstr(inflow - outflow == 1, name=f'FlowBalance_{k}')
    for (i, j) in directed_arcs:
        undirected = arc_to_undirected[i, j]
        m.addConstr(f[i, j] <= M * y[undirected], name=f'FlowLink_{i}_{j}')
    m.optimize()
    if m.Status == gp.GRB.OPTIMAL:
        print(f'ObjVal {m.ObjVal}')
        for e in undirected_edges:
            print(f'{y[e].VarName} {y[e].X}')
        for arc in directed_arcs:
            print(f'{f[arc].VarName} {f[arc].X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_mst_single_commodity_flow()