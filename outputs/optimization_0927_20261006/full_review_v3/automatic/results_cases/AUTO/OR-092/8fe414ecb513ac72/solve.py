import gurobipy as gp
import pandas as pd
import numpy as np
import re
sources_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_sources.csv', dtype=str, keep_default_na=False)
sources_df['source_id'] = sources_df['source_id'].str.strip()
sources_df['supply_units'] = sources_df['supply_units'].astype(int)
sources = list(sources_df['source_id'])
supply_dict = dict(zip(sources_df['source_id'], sources_df['supply_units']))
dest_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_destinations.csv', dtype=str, keep_default_na=False)
dest_df['destination_id'] = dest_df['destination_id'].str.strip()
dest_df['demand_units'] = dest_df['demand_units'].astype(int)
destinations = list(dest_df['destination_id'])
demand_dict = dict(zip(dest_df['destination_id'], dest_df['demand_units']))
cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_cost_matrix.csv', dtype=str, keep_default_na=False)
cost_df['source_id'] = cost_df['source_id'].str.strip()
cost_columns = [col for col in cost_df.columns if col != 'source_id']
cost_columns_norm = [col.strip() for col in cost_columns]
if set(cost_columns_norm) != set(destinations):
    raise ValueError('Destination columns in cost matrix do not match destination IDs from destinations file.')
cost_dict = {}
for (_, row) in cost_df.iterrows():
    src = row['source_id']
    for d in destinations:
        cost_dict[src, d] = float(row[d])
TRUCK_CAPACITY = 10
m = gp.Model('Transportation_MILP')
t_vars = m.addVars(sources, destinations, vtype=gp.GRB.INTEGER, lb=0, name='')
x_vars = m.addVars(sources, destinations, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
m.setObjective(gp.quicksum((cost_dict[i, j] * x_vars[i, j] for i in sources for j in destinations)), gp.GRB.MINIMIZE)
for i in sources:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in destinations)) <= supply_dict[i], name=f'supply_{i}')
for j in destinations:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in sources)) == demand_dict[j], name=f'demand_{j}')
for i in sources:
    for j in destinations:
        m.addConstr(x_vars[i, j] <= TRUCK_CAPACITY * t_vars[i, j], name=f'truckcap_{i}_{j}')
m.optimize()