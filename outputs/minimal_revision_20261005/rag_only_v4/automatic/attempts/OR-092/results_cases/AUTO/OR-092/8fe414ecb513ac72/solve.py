import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
cost_matrix_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_cost_matrix.csv'
destinations_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_destinations.csv'
sources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_sources.csv'
cost_df = pd.read_csv(cost_matrix_path, sep=',')
dest_df = pd.read_csv(destinations_path, sep=',')
src_df = pd.read_csv(sources_path, sep=',')
cost_df['source_id'] = cost_df['source_id'].astype(str).str.strip()
src_df['source_id'] = src_df['source_id'].astype(str).str.strip()
dest_df['destination_id'] = dest_df['destination_id'].astype(str).str.strip()
sources = ['S%d' % i for i in range(1, 11)]
destinations = ['D%d' % i for i in range(1, 21)]
missing_sources = set(sources) - set(src_df['source_id'])
if missing_sources:
    raise ValueError(f'Missing sources in expanded_sources.csv: {missing_sources}')
missing_destinations = set(destinations) - set(dest_df['destination_id'])
if missing_destinations:
    raise ValueError(f'Missing destinations in expanded_destinations.csv: {missing_destinations}')
missing_cost_sources = set(sources) - set(cost_df['source_id'])
if missing_cost_sources:
    raise ValueError(f'Missing sources in expanded_cost_matrix.csv: {missing_cost_sources}')
missing_cost_dest_cols = set(destinations) - set(cost_df.columns[1:])
if missing_cost_dest_cols:
    raise ValueError(f'Missing destination columns in expanded_cost_matrix.csv: {missing_cost_dest_cols}')
supply_units = src_df.set_index('source_id').loc[sources, 'supply_units'].to_dict()
demand_units = dest_df.set_index('destination_id').loc[destinations, 'demand_units'].to_dict()
cost_dict = {}
for s in sources:
    row = cost_df[cost_df['source_id'] == s]
    if row.empty:
        raise ValueError(f'Source {s} missing in cost matrix.')
    for d in destinations:
        val = row.iloc[0][d]
        if pd.isnull(val):
            raise ValueError(f'Missing cost for source {s}, destination {d}.')
        cost_dict[s, d] = float(val)
sd_pairs = [(s, d) for s in sources for d in destinations]

def solve_problem():
    m = gp.Model()
    m.Params.MIPGap = 0.0001
    trucks = m.addVars(sd_pairs, vtype=GRB.INTEGER, lb=0, ub=GRB.INFINITY, name='')
    cargo = m.addVars(sd_pairs, vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name='')
    m.setObjective(gp.quicksum((cost_dict[s, d] * cargo[s, d] for (s, d) in sd_pairs)), GRB.MINIMIZE)
    for s in sources:
        m.addConstr(gp.quicksum((cargo[s, d] for d in destinations)) <= supply_units[s], name='')
    for d in destinations:
        m.addConstr(gp.quicksum((cargo[s, d] for s in sources)) == demand_units[d], name='')
    for (s, d) in sd_pairs:
        m.addConstr(cargo[s, d] <= 10 * trucks[s, d], name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for (s, d) in sd_pairs:
            print(f'{trucks[s, d].VarName} {trucks[s, d].X}')
            print(f'{cargo[s, d].VarName} {cargo[s, d].X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()