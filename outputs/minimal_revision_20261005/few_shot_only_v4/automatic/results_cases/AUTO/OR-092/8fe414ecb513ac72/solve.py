import gurobipy as gp
import pandas as pd
import numpy as np

def solve_problem():
    cost_matrix_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_cost_matrix.csv'
    destinations_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_destinations.csv'
    sources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_sources.csv'
    cost_df = pd.read_csv(cost_matrix_path, sep=',')
    dest_df = pd.read_csv(destinations_path, sep=',')
    src_df = pd.read_csv(sources_path, sep=',')
    sources = [f'S{i}' for i in range(1, 11)]
    destinations = [f'D{i}' for i in range(1, 21)]
    if not set(sources).issubset(set(src_df['source_id'].astype(str))):
        missing = set(sources) - set(src_df['source_id'].astype(str))
        raise ValueError(f'Missing sources in expanded_sources.csv: {missing}')
    if not set(destinations).issubset(set(dest_df['destination_id'].astype(str))):
        missing = set(destinations) - set(dest_df['destination_id'].astype(str))
        raise ValueError(f'Missing destinations in expanded_destinations.csv: {missing}')
    if not set(sources).issubset(set(cost_df['source_id'].astype(str))):
        missing = set(sources) - set(cost_df['source_id'].astype(str))
        raise ValueError(f'Missing sources in expanded_cost_matrix.csv: {missing}')
    if not set(destinations).issubset(set(cost_df.columns[1:])):
        missing = set(destinations) - set(cost_df.columns[1:])
        raise ValueError(f'Missing destinations in expanded_cost_matrix.csv columns: {missing}')
    src_df['source_id'] = src_df['source_id'].astype(str)
    dest_df['destination_id'] = dest_df['destination_id'].astype(str)
    cost_df['source_id'] = cost_df['source_id'].astype(str)
    supply_units = dict(zip(src_df['source_id'], src_df['supply_units']))
    demand_units = dict(zip(dest_df['destination_id'], dest_df['demand_units']))
    cost = {}
    for (_, row) in cost_df.iterrows():
        s = str(row['source_id'])
        for d in destinations:
            cost[s, d] = float(row[d])
    keys = [(s, d) for s in sources for d in destinations]
    m = gp.Model('TruckTransportation')
    m.Params.MIPGap = 0.0001
    t = m.addVars(keys, vtype=gp.GRB.INTEGER, lb=0, name='')
    f = m.addVars(keys, vtype=gp.GRB.CONTINUOUS, lb=0, name='')
    m.setObjective(gp.quicksum((cost[s, d] * f[s, d] for (s, d) in keys)), gp.GRB.MINIMIZE)
    for s in sources:
        m.addConstr(gp.quicksum((f[s, d] for d in destinations)) <= float(supply_units[s]), name='supply_' + s)
    for d in destinations:
        m.addConstr(gp.quicksum((f[s, d] for s in sources)) == float(demand_units[d]), name='demand_' + d)
    for (s, d) in keys:
        m.addConstr(f[s, d] <= 10 * t[s, d], name='truckcap_%s_%s' % (s, d))
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        print(f'ObjVal {m.objVal}')
        for (s, d) in keys:
            print(f'{t[s, d].VarName} {t[s, d].X}')
            print(f'{f[s, d].VarName} {f[s, d].X}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_problem()