import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
cost_matrix_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_cost_matrix.csv'
destinations_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_destinations.csv'
sources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_sources.csv'
sources_df = pd.read_csv(sources_path, dtype=str, keep_default_na=False)
sources_df['supply_units'] = sources_df['supply_units'].astype(int)
sources_df['source_id'] = sources_df['source_id'].str.strip()
sources_list = ['S%d' % i for i in range(1, 11)]
sources_df = sources_df[sources_df['source_id'].isin(sources_list)].copy()
if set(sources_list) != set(sources_df['source_id']):
    missing = set(sources_list) - set(sources_df['source_id'])
    raise ValueError(f'Missing sources in CSV: {missing}')
supply_dict = dict(zip(sources_df['source_id'], sources_df['supply_units']))
dest_df = pd.read_csv(destinations_path, dtype=str, keep_default_na=False)
dest_df['demand_units'] = dest_df['demand_units'].astype(int)
dest_df['destination_id'] = dest_df['destination_id'].str.strip()
dest_list = ['D%d' % i for i in range(1, 21)]
dest_df = dest_df[dest_df['destination_id'].isin(dest_list)].copy()
if set(dest_list) != set(dest_df['destination_id']):
    missing = set(dest_list) - set(dest_df['destination_id'])
    raise ValueError(f'Missing destinations in CSV: {missing}')
demand_dict = dict(zip(dest_df['destination_id'], dest_df['demand_units']))
cost_df = pd.read_csv(cost_matrix_path, dtype=str, keep_default_na=False)
cost_df['source_id'] = cost_df['source_id'].str.strip()
cost_df = cost_df[cost_df['source_id'].isin(sources_list)].copy()
cost_df = cost_df.set_index('source_id')
for d in dest_list:
    cost_df[d] = cost_df[d].astype(float)
if not all((d in cost_df.columns for d in dest_list)):
    missing = set(dest_list) - set(cost_df.columns)
    raise ValueError(f'Missing destination columns in cost matrix: {missing}')
if not all((s in cost_df.index for s in sources_list)):
    missing = set(sources_list) - set(cost_df.index)
    raise ValueError(f'Missing source rows in cost matrix: {missing}')
cost_dict = {(s, d): cost_df.at[s, d] for s in sources_list for d in dest_list}
S = sources_list
D = dest_list
truck_capacity = 10
m = Model('transportation_mip')
m.Params.OutputFlag = 0
t_vars = m.addVars(S, D, vtype=GRB.INTEGER, lb=0, name='')
x_vars = m.addVars(S, D, vtype=GRB.CONTINUOUS, lb=0, name='')
for s in S:
    m.addConstr(quicksum((x_vars[s, d] for d in D)) <= supply_dict[s])
for d in D:
    m.addConstr(quicksum((x_vars[s, d] for s in S)) == demand_dict[d])
for s in S:
    for d in D:
        m.addConstr(x_vars[s, d] <= truck_capacity * t_vars[s, d])
m.setObjective(quicksum((cost_dict[s, d] * x_vars[s, d] for s in S for d in D)), GRB.MINIMIZE)
m.optimize()