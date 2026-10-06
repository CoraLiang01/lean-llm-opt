import gurobipy as gp
import pandas as pd
import numpy as np
cost_matrix_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_cost_matrix.csv'
destinations_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_destinations.csv'
sources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_sources.csv'
cost_df = pd.read_csv(cost_matrix_path, sep=',')
dest_df = pd.read_csv(destinations_path, sep=',')
src_df = pd.read_csv(sources_path, sep=',')
cost_df['source_id'] = cost_df['source_id'].astype(str).str.strip()
src_df['source_id'] = src_df['source_id'].astype(str).str.strip()
dest_df['destination_id'] = dest_df['destination_id'].astype(str).str.strip()
sources = list(src_df['source_id'])
destinations = list(dest_df['destination_id'])
cost_sources = set(cost_df['source_id'])
cost_destinations = set(cost_df.columns) - {'source_id'}
if set(sources) - cost_sources:
    raise ValueError(f'Cost matrix missing sources: {set(sources) - cost_sources}')
if set(destinations) - cost_destinations:
    raise ValueError(f'Cost matrix missing destinations: {set(destinations) - cost_destinations}')
cost = {}
for _, row in cost_df.iterrows():
    i = row['source_id']
    for j in destinations:
        cost[i, j] = float(row[j])
supply = dict(zip(src_df['source_id'], src_df['supply_units']))
demand = dict(zip(dest_df['destination_id'], dest_df['demand_units']))
truck_capacity = 10
m = gp.Model('TransportationWithIntegerTrucks')
t = m.addVars(sources, destinations, vtype=gp.GRB.INTEGER, lb=0, name='')
x = m.addVars(sources, destinations, vtype=gp.GRB.CONTINUOUS, lb=0, name='')
m.setObjective(gp.quicksum((cost[i, j] * x[i, j] for i in sources for j in destinations)), gp.GRB.MINIMIZE)
for i in sources:
    m.addConstr(gp.quicksum((x[i, j] for j in destinations)) <= supply[i], name=f'supply_{i}')
for j in destinations:
    m.addConstr(gp.quicksum((x[i, j] for i in sources)) == demand[j], name=f'demand_{j}')
for i in sources:
    for j in destinations:
        m.addConstr(x[i, j] <= truck_capacity * t[i, j], name=f'truckcap_{i}_{j}')
m.optimize()