import gurobipy as gp
import pandas as pd
import numpy as np
import re
cost_matrix_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_cost_matrix.csv'
sources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_sources.csv'
destinations_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_destinations.csv'
cost_df = pd.read_csv(cost_matrix_path, dtype=str, keep_default_na=False)
sources_df = pd.read_csv(sources_path, dtype=str, keep_default_na=False)
dest_df = pd.read_csv(destinations_path, dtype=str, keep_default_na=False)
sources_df['supply_units'] = sources_df['supply_units'].astype(float)
sources = list(sources_df['source_id'])
supply_dict = dict(zip(sources_df['source_id'], sources_df['supply_units']))
dest_df['demand_units'] = dest_df['demand_units'].astype(float)
destinations = list(dest_df['destination_id'])
demand_dict = dict(zip(dest_df['destination_id'], dest_df['demand_units']))
cost_dict = {}
for (_, row) in cost_df.iterrows():
    s = row['source_id']
    for d in destinations:
        val = row[d]
        try:
            cost = float(val)
        except Exception:
            raise ValueError(f"Missing or invalid cost for source {s}, destination {d}: '{val}'")
        cost_dict[s, d] = cost
for s in sources:
    if s not in supply_dict:
        raise KeyError(f'Source {s} missing in supply data')
for d in destinations:
    if d not in demand_dict:
        raise KeyError(f'Destination {d} missing in demand data')
for s in sources:
    for d in destinations:
        if (s, d) not in cost_dict:
            raise KeyError(f'Missing cost for source {s}, destination {d}')
TRUCK_CAPACITY = 10.0
m = gp.Model('TruckTransportation')
t_vars = m.addVars(sources, destinations, vtype=gp.GRB.INTEGER, lb=0, name='')
q_vars = m.addVars(sources, destinations, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
m.setObjective(gp.quicksum((cost_dict[s, d] * q_vars[s, d] for s in sources for d in destinations)), gp.GRB.MINIMIZE)
for s in sources:
    m.addConstr(gp.quicksum((q_vars[s, d] for d in destinations)) <= supply_dict[s], name=f'supply_{s}')
for d in destinations:
    m.addConstr(gp.quicksum((q_vars[s, d] for s in sources)) == demand_dict[d], name=f'demand_{d}')
for s in sources:
    for d in destinations:
        m.addConstr(q_vars[s, d] <= TRUCK_CAPACITY * t_vars[s, d], name=f'truckcap_{s}_{d}')
m.optimize()