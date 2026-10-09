import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
cost_matrix_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_cost_matrix.csv'
sources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_sources.csv'
destinations_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_destinations.csv'
sources_df = pd.read_csv(sources_path, dtype=str, keep_default_na=False)
sources_df['supply_units'] = sources_df['supply_units'].astype(int)
sources_df['source_id'] = sources_df['source_id'].astype(str)
sources = list(sources_df['source_id'])
destinations_df = pd.read_csv(destinations_path, dtype=str, keep_default_na=False)
destinations_df['demand_units'] = destinations_df['demand_units'].astype(int)
destinations_df['destination_id'] = destinations_df['destination_id'].astype(str)
destinations = list(destinations_df['destination_id'])
cost_matrix_df = pd.read_csv(cost_matrix_path, dtype=str, keep_default_na=False)
cost_matrix_df['source_id'] = cost_matrix_df['source_id'].astype(str)
for d in destinations:
    cost_matrix_df[d] = cost_matrix_df[d].astype(float)
cost_matrix_sources = set(cost_matrix_df['source_id'])
if set(sources) != cost_matrix_sources:
    raise ValueError(f'Mismatch between sources in sources file and cost matrix: {set(sources) ^ cost_matrix_sources}')
cost_matrix_destinations = set(cost_matrix_df.columns) - {'source_id'}
if set(destinations) != cost_matrix_destinations:
    raise ValueError(f'Mismatch between destinations in destinations file and cost matrix: {set(destinations) ^ cost_matrix_destinations}')
cost = {}
for (_, row) in cost_matrix_df.iterrows():
    s = row['source_id']
    for d in destinations:
        cost[s, d] = float(row[d])
supply_units = dict(zip(sources_df['source_id'], sources_df['supply_units']))
demand_units = dict(zip(destinations_df['destination_id'], destinations_df['demand_units']))
truck_capacity = 10
m = Model('transportation_truck_integer')
x_vars = m.addVars(sources, destinations, lb=0.0, vtype=GRB.CONTINUOUS, name='')
t_vars = m.addVars(sources, destinations, lb=0, vtype=GRB.INTEGER, name='')
for s in sources:
    m.addConstr(quicksum((x_vars[s, d] for d in destinations)) <= supply_units[s], name='')
for d in destinations:
    m.addConstr(quicksum((x_vars[s, d] for s in sources)) == demand_units[d], name='')
for s in sources:
    for d in destinations:
        m.addConstr(x_vars[s, d] <= truck_capacity * t_vars[s, d], name='')
m.setObjective(quicksum((cost[s, d] * x_vars[s, d] for s in sources for d in destinations)), GRB.MINIMIZE)
m.optimize()