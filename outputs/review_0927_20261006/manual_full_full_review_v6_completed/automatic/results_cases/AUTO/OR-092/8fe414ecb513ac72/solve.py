import gurobipy as gp
import pandas as pd
import numpy as np
sources_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_sources.csv', dtype=str, keep_default_na=False)
if 'source_id' not in sources_df.columns or 'supply_units' not in sources_df.columns:
    raise KeyError("expanded_sources.csv must contain 'source_id' and 'supply_units' columns.")
sources_df['source_id'] = sources_df['source_id'].str.strip()
sources_df['supply_units'] = sources_df['supply_units'].astype(int)
sources = list(sources_df['source_id'])
supply_dict = dict(zip(sources_df['source_id'], sources_df['supply_units']))
dest_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_destinations.csv', dtype=str, keep_default_na=False)
if 'destination_id' not in dest_df.columns or 'demand_units' not in dest_df.columns:
    raise KeyError("expanded_destinations.csv must contain 'destination_id' and 'demand_units' columns.")
dest_df['destination_id'] = dest_df['destination_id'].str.strip()
dest_df['demand_units'] = dest_df['demand_units'].astype(int)
destinations = list(dest_df['destination_id'])
demand_dict = dict(zip(dest_df['destination_id'], dest_df['demand_units']))
cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_cost_matrix.csv', dtype=str, keep_default_na=False)
if 'source_id' not in cost_df.columns:
    raise KeyError("expanded_cost_matrix.csv must contain 'source_id' column.")
cost_df['source_id'] = cost_df['source_id'].str.strip()
for d in destinations:
    if d not in cost_df.columns:
        raise KeyError(f"expanded_cost_matrix.csv missing destination column '{d}'.")
cost_dict = {}
for _, row in cost_df.iterrows():
    s = row['source_id']
    for d in destinations:
        try:
            cost_dict[s, d] = float(row[d])
        except Exception as e:
            raise ValueError(f'Invalid cost value for source {s}, destination {d}: {row[d]}') from e
TRUCK_CAPACITY = 10
m = gp.Model('Transportation_MILP')
t_vars = m.addVars(sources, destinations, vtype=gp.GRB.INTEGER, lb=0, name='')
x_vars = m.addVars(sources, destinations, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
for s in sources:
    m.addConstr(gp.quicksum((x_vars[s, d] for d in destinations)) <= supply_dict[s], name=f'supply_{s}')
for d in destinations:
    m.addConstr(gp.quicksum((x_vars[s, d] for s in sources)) == demand_dict[d], name=f'demand_{d}')
for s in sources:
    for d in destinations:
        m.addConstr(x_vars[s, d] <= TRUCK_CAPACITY * t_vars[s, d], name=f'truckload_{s}_{d}')
m.setObjective(gp.quicksum((cost_dict[s, d] * x_vars[s, d] for s in sources for d in destinations)), gp.GRB.MINIMIZE)
m.optimize()