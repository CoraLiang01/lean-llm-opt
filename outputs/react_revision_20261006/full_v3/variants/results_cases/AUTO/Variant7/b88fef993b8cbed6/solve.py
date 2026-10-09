import gurobipy as gp
import pandas as pd
import numpy as np
supply_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant7/inputs/node_supply_demand.csv'
hub_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant7/inputs/hub_capacity.csv'
arc_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant7/inputs/arc_costs.csv'

def solve_transshipment():
    df_nodes = pd.read_csv(supply_demand_path, sep=',')
    df_nodes['NodeType'] = df_nodes['NodeType'].str.strip().str.casefold()
    df_nodes['Node'] = df_nodes['Node'].astype(str).str.strip()
    sources = df_nodes.loc[df_nodes['NodeType'] == 'sourcesupply', 'Node'].tolist()
    customers = df_nodes.loc[df_nodes['NodeType'] == 'customerdemand', 'Node'].tolist()
    supply = df_nodes.set_index('Node').loc[sources, 'Amount'].to_dict()
    demand = df_nodes.set_index('Node').loc[customers, 'Amount'].to_dict()
    df_hubs = pd.read_csv(hub_capacity_path, sep=',')
    df_hubs['Hub'] = df_hubs['Hub'].astype(str).str.strip()
    hubs = df_hubs['Hub'].tolist()
    hub_capacity = df_hubs.set_index('Hub')['ThroughputCapacity'].to_dict()
    df_arcs = pd.read_csv(arc_costs_path, sep=',')
    df_arcs['From'] = df_arcs['From'].astype(str).str.strip()
    df_arcs['To'] = df_arcs['To'].astype(str).str.strip()
    arcs = [(row['From'], row['To']) for (_, row) in df_arcs.iterrows()]
    arc_cost = {(row['From'], row['To']): row['Cost'] for (_, row) in df_arcs.iterrows()}
    for (i, j) in arcs:
        if (i, j) not in arc_cost:
            raise ValueError(f'Missing cost for arc ({i}, {j})')
    for s in sources:
        if not any((i == s for (i, j) in arcs)):
            raise ValueError(f'Source {s} has no outgoing arcs in arc_costs.csv')
    for c in customers:
        if not any((j == c for (i, j) in arcs)):
            raise ValueError(f'Customer {c} has no incoming arcs in arc_costs.csv')
    for h in hubs:
        if not any((j == h for (i, j) in arcs)):
            raise ValueError(f'Hub {h} has no incoming arcs in arc_costs.csv')
        if not any((i == h for (i, j) in arcs)):
            raise ValueError(f'Hub {h} has no outgoing arcs in arc_costs.csv')
    m = gp.Model('min_cost_transshipment')
    m.Params.MIPGap = 0.0001
    f = m.addVars(arcs, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((arc_cost[i, j] * f[i, j] for (i, j) in arcs)), gp.GRB.MINIMIZE)
    for s in sources:
        out_arcs = [(i, j) for (i, j) in arcs if i == s]
        m.addConstr(gp.quicksum((f[i, j] for (i, j) in out_arcs)) <= supply[s], name=f'supply_{s}')
    for c in customers:
        in_arcs = [(i, j) for (i, j) in arcs if j == c]
        m.addConstr(gp.quicksum((f[i, j] for (i, j) in in_arcs)) >= demand[c], name=f'demand_{c}')
    for h in hubs:
        in_arcs = [(i, j) for (i, j) in arcs if j == h]
        out_arcs = [(i, j) for (i, j) in arcs if i == h]
        m.addConstr(gp.quicksum((f[i, j] for (i, j) in in_arcs)) == gp.quicksum((f[i, j] for (i, j) in out_arcs)), name=f'flowbal_{h}')
    for h in hubs:
        in_arcs = [(i, j) for (i, j) in arcs if j == h]
        m.addConstr(gp.quicksum((f[i, j] for (i, j) in in_arcs)) <= hub_capacity[h], name=f'cap_{h}')
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        print(f'ObjVal {m.ObjVal}')
        for (i, j) in arcs:
            var = f[i, j]
            print(f'{var.VarName} {var.X}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_transshipment()