import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
cost_matrix_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_cost_matrix.csv'
sources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_sources.csv'
destinations_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_destinations.csv'
sources_df = pd.read_csv(sources_path, dtype=str, keep_default_na=False)
if 'source_id' not in sources_df.columns or 'supply_units' not in sources_df.columns:
    raise ValueError('expanded_sources.csv missing required columns')
sources_df['supply_units'] = sources_df['supply_units'].astype(int)
sources = sources_df['source_id'].tolist()
supply_units = dict(zip(sources_df['source_id'], sources_df['supply_units']))
dest_df = pd.read_csv(destinations_path, dtype=str, keep_default_na=False)
if 'destination_id' not in dest_df.columns or 'demand_units' not in dest_df.columns:
    raise ValueError('expanded_destinations.csv missing required columns')
dest_df['demand_units'] = dest_df['demand_units'].astype(int)
destinations = dest_df['destination_id'].tolist()
demand_units = dict(zip(dest_df['destination_id'], dest_df['demand_units']))
cost_df = pd.read_csv(cost_matrix_path, dtype=str, keep_default_na=False)
if 'source_id' not in cost_df.columns:
    raise ValueError("expanded_cost_matrix.csv missing 'source_id' column")
cost_df.set_index('source_id', inplace=True)
for d in destinations:
    if d not in cost_df.columns:
        raise ValueError(f'Destination {d} missing in cost matrix columns')
for d in destinations:
    cost_df[d] = cost_df[d].astype(float)
for s in sources:
    if s not in cost_df.index:
        raise ValueError(f'Source {s} missing in cost matrix rows')
cost = {(s, d): cost_df.at[s, d] for s in sources for d in destinations}
truck_capacity = 10
m = Model('transportation_mip')
trucks_vars = m.addVars(sources, destinations, vtype=GRB.INTEGER, lb=0, name='')
cargo_vars = m.addVars(sources, destinations, vtype=GRB.CONTINUOUS, lb=0, name='')
for s in sources:
    m.addConstr(quicksum((cargo_vars[s, d] for d in destinations)) <= supply_units[s])
for d in destinations:
    m.addConstr(quicksum((cargo_vars[s, d] for s in sources)) == demand_units[d])
for s in sources:
    for d in destinations:
        m.addConstr(cargo_vars[s, d] <= truck_capacity * trucks_vars[s, d])
m.setObjective(quicksum((cost[s, d] * cargo_vars[s, d] for s in sources for d in destinations)), GRB.MINIMIZE)
m.optimize()