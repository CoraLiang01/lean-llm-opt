import gurobipy as gp
import pandas as pd
import numpy as np
cost_matrix_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_cost_matrix.csv'
destinations_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_destinations.csv'
sources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_sources.csv'

def solve_problem():
    cost_df = pd.read_csv(cost_matrix_path, sep=',')
    dest_df = pd.read_csv(destinations_path, sep=',')
    src_df = pd.read_csv(sources_path, sep=',')
    sources = [f'S{i}' for i in range(1, 11)]
    destinations = [f'D{i}' for i in range(1, 21)]
    cost_sources = set(cost_df['source_id'].astype(str))
    if set(sources) - cost_sources:
        raise ValueError(f'Missing sources in cost matrix: {set(sources) - cost_sources}')
    cost_dest_cols = set(cost_df.columns) - {'source_id'}
    if set(destinations) - cost_dest_cols:
        raise ValueError(f'Missing destinations in cost matrix: {set(destinations) - cost_dest_cols}')
    src_ids = set(src_df['source_id'].astype(str))
    if set(sources) - src_ids:
        raise ValueError(f'Missing sources in sources.csv: {set(sources) - src_ids}')
    dest_ids = set(dest_df['destination_id'].astype(str))
    if set(destinations) - dest_ids:
        raise ValueError(f'Missing destinations in destinations.csv: {set(destinations) - dest_ids}')
    cost_dict = {}
    for (_, row) in cost_df.iterrows():
        s = str(row['source_id'])
        for d in destinations:
            cost = row[d]
            if pd.isnull(cost):
                raise ValueError(f'Missing cost for ({s},{d}) in cost matrix')
            cost_dict[s, d] = float(cost)
    supply_dict = {}
    for (_, row) in src_df.iterrows():
        s = str(row['source_id'])
        supply = row['supply_units']
        if pd.isnull(supply):
            raise ValueError(f'Missing supply for source {s}')
        supply_dict[s] = float(supply)
    demand_dict = {}
    for (_, row) in dest_df.iterrows():
        d = str(row['destination_id'])
        demand = row['demand_units']
        if pd.isnull(demand):
            raise ValueError(f'Missing demand for destination {d}')
        demand_dict[d] = float(demand)
    keys = [(s, d) for s in sources for d in destinations]
    m = gp.Model('Truck_Dispatch')
    m.setParam('MIPGap', 0.0001)
    t = m.addVars(keys, vtype=gp.GRB.INTEGER, lb=0, name='')
    f = m.addVars(keys, vtype=gp.GRB.CONTINUOUS, lb=0, name='')
    m.setObjective(gp.quicksum((cost_dict[s, d] * f[s, d] for (s, d) in keys)), gp.GRB.MINIMIZE)
    for s in sources:
        m.addConstr(gp.quicksum((f[s, d] for d in destinations)) <= supply_dict[s], name='supply_' + s)
    for d in destinations:
        m.addConstr(gp.quicksum((f[s, d] for s in sources)) == demand_dict[d], name='demand_' + d)
    for (s, d) in keys:
        m.addConstr(f[s, d] <= 10 * t[s, d], name='truckload_%s_%s' % (s, d))
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        print(f'ObjVal: {m.objVal}')
        for (s, d) in keys:
            print(f't[{s},{d}] {t[s, d].VarName} {t[s, d].X}')
            print(f'f[{s},{d}] {f[s, d].VarName} {f[s, d].X}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_problem()