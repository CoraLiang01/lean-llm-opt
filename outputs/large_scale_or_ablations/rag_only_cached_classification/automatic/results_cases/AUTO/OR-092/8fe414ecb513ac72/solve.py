import pandas as pd
import numpy as np
from gurobipy import Model, GRB
sources_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_sources.csv', sep=',')
sources_df['source_id'] = sources_df['source_id'].astype(str)
sources = list(sources_df['source_id'])
dest_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_destinations.csv', sep=',')
dest_df['destination_id'] = dest_df['destination_id'].astype(str)
destinations = list(dest_df['destination_id'])
cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_cost_matrix.csv', sep=',')
cost_df['source_id'] = cost_df['source_id'].astype(str)
cost_sources = set(cost_df['source_id'])
if set(sources) != cost_sources:
    raise ValueError(f'Mismatch between sources in sources.csv and cost_matrix.csv: {set(sources) ^ cost_sources}')
cost_dest_cols = [col for col in cost_df.columns if col != 'source_id']
if set(destinations) != set(cost_dest_cols):
    raise ValueError(f'Mismatch between destinations in destinations.csv and cost_matrix.csv: {set(destinations) ^ set(cost_dest_cols)}')
cost = {}
for _, row in cost_df.iterrows():
    s = row['source_id']
    cost[s] = {}
    for d in destinations:
        cost[s][d] = float(row[d])
supply = dict(zip(sources_df['source_id'], sources_df['supply_units']))
demand = dict(zip(dest_df['destination_id'], dest_df['demand_units']))
truck_capacity = 10
m = Model('transportation_milp')
t = m.addVars(sources, destinations, vtype=GRB.INTEGER, lb=0, name='')
q = m.addVars(sources, destinations, vtype=GRB.CONTINUOUS, lb=0, name='')
for s in sources:
    for d in destinations:
        m.addConstr(q[s, d] <= truck_capacity * t[s, d])
for s in sources:
    m.addConstr(sum((q[s, d] for d in destinations)) <= supply[s])
for d in destinations:
    m.addConstr(sum((q[s, d] for s in sources)) == demand[d])
m.setObjective(sum((cost[s][d] * q[s, d] for s in sources for d in destinations)), GRB.MINIMIZE)
m.optimize()