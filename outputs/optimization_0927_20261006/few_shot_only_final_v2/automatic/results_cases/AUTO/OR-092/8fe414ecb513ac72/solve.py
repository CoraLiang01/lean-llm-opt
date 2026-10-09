import gurobipy as gp
import pandas as pd
import numpy as np
import re
cost_matrix_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_cost_matrix.csv'
destinations_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_destinations.csv'
sources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_sources.csv'
cost_df = pd.read_csv(cost_matrix_path, dtype=str, keep_default_na=False)
dest_df = pd.read_csv(destinations_path, dtype=str, keep_default_na=False)
src_df = pd.read_csv(sources_path, dtype=str, keep_default_na=False)
sources = [f'S{i}' for i in range(1, 11)]
destinations = [f'D{i}' for i in range(1, 21)]
cost_per_unit = {}
for (_, row) in cost_df.iterrows():
    s = str(row['source_id'])
    if s not in sources:
        continue
    for d in destinations:
        if d not in row:
            raise KeyError(f'Destination {d} missing in cost matrix for source {s}')
        try:
            cost_per_unit[s, d] = float(row[d])
        except ValueError:
            raise ValueError(f'Non-numeric cost for route ({s},{d}): {row[d]}')
supply_units = {}
for (_, row) in src_df.iterrows():
    s = str(row['source_id'])
    if s not in sources:
        continue
    try:
        supply_units[s] = float(row['supply_units'])
    except ValueError:
        raise ValueError(f"Non-numeric supply_units for source {s}: {row['supply_units']}")
demand_units = {}
for (_, row) in dest_df.iterrows():
    d = str(row['destination_id'])
    if d not in destinations:
        continue
    try:
        demand_units[d] = float(row['demand_units'])
    except ValueError:
        raise ValueError(f"Non-numeric demand_units for destination {d}: {row['demand_units']}")
if set(sources) != set(supply_units.keys()):
    missing = set(sources) - set(supply_units.keys())
    raise KeyError(f'Missing supply data for sources: {missing}')
if set(destinations) != set(demand_units.keys()):
    missing = set(destinations) - set(demand_units.keys())
    raise KeyError(f'Missing demand data for destinations: {missing}')
for s in sources:
    for d in destinations:
        if (s, d) not in cost_per_unit:
            raise KeyError(f'Missing cost for route ({s},{d})')
TRUCK_CAPACITY = 10.0
m = gp.Model('Transportation_MILP')
t_vars = m.addVars(sources, destinations, vtype=gp.GRB.INTEGER, lb=0, name='')
q_vars = m.addVars(sources, destinations, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
m.setObjective(gp.quicksum((cost_per_unit[s, d] * q_vars[s, d] for s in sources for d in destinations)), gp.GRB.MINIMIZE)
for s in sources:
    m.addConstr(gp.quicksum((q_vars[s, d] for d in destinations)) <= supply_units[s], name=f'supply_{s}')
for d in destinations:
    m.addConstr(gp.quicksum((q_vars[s, d] for s in sources)) == demand_units[d], name=f'demand_{d}')
for s in sources:
    for d in destinations:
        m.addConstr(q_vars[s, d] <= TRUCK_CAPACITY * t_vars[s, d], name=f'truckload_{s}_{d}')
m.optimize()