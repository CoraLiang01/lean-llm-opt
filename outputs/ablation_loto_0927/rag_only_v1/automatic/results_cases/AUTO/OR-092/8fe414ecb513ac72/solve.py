import pandas as pd
import numpy as np
from gurobipy import Model, GRB
sources_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_sources.csv', sep=',')
sources_df['source_id'] = sources_df['source_id'].astype(str).str.strip()
sources = list(sources_df['source_id'])
dest_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_destinations.csv', sep=',')
dest_df['destination_id'] = dest_df['destination_id'].astype(str).str.strip()
destinations = list(dest_df['destination_id'])
cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_cost_matrix.csv', sep=',')
cost_df['source_id'] = cost_df['source_id'].astype(str).str.strip()
missing_sources = set(sources) - set(cost_df['source_id'])
if missing_sources:
    raise ValueError(f'Missing sources in cost matrix: {missing_sources}')
missing_destinations = set(destinations) - set([c for c in cost_df.columns if c != 'source_id'])
if missing_destinations:
    raise ValueError(f'Missing destinations in cost matrix: {missing_destinations}')
cost = {}
for (_, row) in cost_df.iterrows():
    s = str(row['source_id']).strip()
    cost[s] = {}
    for d in destinations:
        cost[s][d] = float(row[d])
supply = {}
for (_, row) in sources_df.iterrows():
    s = str(row['source_id']).strip()
    supply[s] = int(row['supply_units'])
demand = {}
for (_, row) in dest_df.iterrows():
    d = str(row['destination_id']).strip()
    demand[d] = int(row['demand_units'])
truck_capacity = 10
m = Model('transportation_truck_integer')
cargo = m.addVars(sources, destinations, lb=0, vtype=GRB.CONTINUOUS, name='')
trucks = m.addVars(sources, destinations, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(sum((cost[s][d] * cargo[s, d] for s in sources for d in destinations)), GRB.MINIMIZE)
for s in sources:
    m.addConstr(sum((cargo[s, d] for d in destinations)) <= supply[s], name=f'supply_{s}')
for d in destinations:
    m.addConstr(sum((cargo[s, d] for s in sources)) == demand[d], name=f'demand_{d}')
for s in sources:
    for d in destinations:
        m.addConstr(cargo[s, d] <= truck_capacity * trucks[s, d], name=f'truckload_{s}_{d}')
m.optimize()