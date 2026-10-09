import gurobipy as gp
import pandas as pd
import numpy as np
cost_matrix_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_cost_matrix.csv'
sources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_sources.csv'
destinations_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_destinations.csv'
cost_df = pd.read_csv(cost_matrix_path, sep=',')
sources_df = pd.read_csv(sources_path, sep=',')
dest_df = pd.read_csv(destinations_path, sep=',')
sources_df['source_id'] = sources_df['source_id'].astype(str).str.strip()
cost_df['source_id'] = cost_df['source_id'].astype(str).str.strip()
dest_df['destination_id'] = dest_df['destination_id'].astype(str).str.strip()
sources = [f'S{i}' for i in range(1, 11)]
destinations = [f'D{i}' for i in range(1, 21)]
missing_sources = set(sources) - set(sources_df['source_id'])
if missing_sources:
    raise ValueError(f'Missing sources in expanded_sources.csv: {missing_sources}')
missing_sources_cost = set(sources) - set(cost_df['source_id'])
if missing_sources_cost:
    raise ValueError(f'Missing sources in expanded_cost_matrix.csv: {missing_sources_cost}')
missing_dest = set(destinations) - set(dest_df['destination_id'])
if missing_dest:
    raise ValueError(f'Missing destinations in expanded_destinations.csv: {missing_dest}')
missing_dest_cost = set(destinations) - set(cost_df.columns[1:])
if missing_dest_cost:
    raise ValueError(f'Missing destinations in expanded_cost_matrix.csv columns: {missing_dest_cost}')
supply_units = dict(zip(sources_df['source_id'], sources_df['supply_units']))
demand_units = dict(zip(dest_df['destination_id'], dest_df['demand_units']))
cost = {}
for (_, row) in cost_df.iterrows():
    s = row['source_id']
    for d in destinations:
        cost[s, d] = float(row[d])
route_keys = [(s, d) for s in sources for d in destinations]
m = gp.Model('Transportation_MILP')
t = m.addVars(route_keys, vtype=gp.GRB.INTEGER, lb=0, name='')
x = m.addVars(route_keys, vtype=gp.GRB.CONTINUOUS, lb=0, name='')
m.setObjective(gp.quicksum((cost[s, d] * x[s, d] for (s, d) in route_keys)), gp.GRB.MINIMIZE)
for s in sources:
    m.addConstr(gp.quicksum((x[s, d] for d in destinations)) <= supply_units[s], name=f'supply_{s}')
for d in destinations:
    m.addConstr(gp.quicksum((x[s, d] for s in sources)) == demand_units[d], name=f'demand_{d}')
for (s, d) in route_keys:
    m.addConstr(x[s, d] <= 10 * t[s, d], name=f'truckcap_{s}_{d}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal {m.objVal}')
    for (s, d) in route_keys:
        print(f'{x[s, d].VarName} {x[s, d].X}')
    for (s, d) in route_keys:
        print(f'{t[s, d].VarName} {t[s, d].X}')
else:
    print(f'Solver status: {m.status}')