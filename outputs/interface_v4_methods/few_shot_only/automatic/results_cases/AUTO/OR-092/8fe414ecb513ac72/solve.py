import gurobipy as gp
import pandas as pd
import numpy as np
cost_matrix_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_cost_matrix.csv'
sources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_sources.csv'
destinations_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_destinations.csv'
cost_df = pd.read_csv(cost_matrix_path, sep=',')
sources_df = pd.read_csv(sources_path, sep=',')
dest_df = pd.read_csv(destinations_path, sep=',')
cost_df['source_id'] = cost_df['source_id'].astype(str).str.strip()
sources_df['source_id'] = sources_df['source_id'].astype(str).str.strip()
dest_df['destination_id'] = dest_df['destination_id'].astype(str).str.strip()
sources = [f'S{i}' for i in range(1, 11)]
destinations = [f'D{j}' for j in range(1, 21)]
missing_sources = set(sources) - set(cost_df['source_id']) - set(sources_df['source_id'])
if missing_sources:
    raise ValueError(f'Missing sources in data: {missing_sources}')
missing_destinations = set(destinations) - set(cost_df.columns[1:]) - set(dest_df['destination_id'])
if missing_destinations:
    raise ValueError(f'Missing destinations in data: {missing_destinations}')
cost = {}
for i in sources:
    row = cost_df.loc[cost_df['source_id'] == i]
    if row.empty:
        raise ValueError(f'Source {i} not found in cost matrix.')
    for j in destinations:
        if j not in cost_df.columns:
            raise ValueError(f'Destination {j} not found in cost matrix columns.')
        cost[i, j] = float(row.iloc[0][j])
supply_units = {}
for i in sources:
    row = sources_df.loc[sources_df['source_id'] == i]
    if row.empty:
        raise ValueError(f'Source {i} not found in sources file.')
    supply_units[i] = int(row.iloc[0]['supply_units'])
demand_units = {}
for j in destinations:
    row = dest_df.loc[dest_df['destination_id'] == j]
    if row.empty:
        raise ValueError(f'Destination {j} not found in destinations file.')
    demand_units[j] = int(row.iloc[0]['demand_units'])
TRUCK_CAPACITY = 10

def solve_transportation():
    m = gp.Model('TruckTransportation')
    x = m.addVars(sources, destinations, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    t = m.addVars(sources, destinations, lb=0, vtype=gp.GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((cost[i, j] * x[i, j] for i in sources for j in destinations)), gp.GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for j in destinations)) <= supply_units[i] for i in sources), name='')
    m.addConstrs((gp.quicksum((x[i, j] for i in sources)) == demand_units[j] for j in destinations), name='')
    m.addConstrs((x[i, j] <= TRUCK_CAPACITY * t[i, j] for i in sources for j in destinations), name='')
    m.optimize()
    return m
m = solve_transportation()