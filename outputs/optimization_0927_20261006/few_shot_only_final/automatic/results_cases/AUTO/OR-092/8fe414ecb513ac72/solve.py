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
if not set(sources).issubset(set(cost_df['source_id'])):
    missing = set(sources) - set(cost_df['source_id'])
    raise ValueError(f'Missing sources in cost matrix: {missing}')
if not set(destinations).issubset(set(cost_df.columns)):
    missing = set(destinations) - set(cost_df.columns)
    raise ValueError(f'Missing destinations in cost matrix: {missing}')
if not set(sources).issubset(set(src_df['source_id'])):
    missing = set(sources) - set(src_df['source_id'])
    raise ValueError(f'Missing sources in sources file: {missing}')
if not set(destinations).issubset(set(dest_df['destination_id'])):
    missing = set(destinations) - set(dest_df['destination_id'])
    raise ValueError(f'Missing destinations in destinations file: {missing}')
cost = {}
for (_, row) in cost_df.iterrows():
    s = row['source_id']
    if s not in sources:
        continue
    cost[s] = {}
    for d in destinations:
        val = row[d]
        try:
            cost[s][d] = float(val)
        except Exception:
            raise ValueError(f'Invalid cost value for source {s}, destination {d}: {val}')
supply_units = {}
for (_, row) in src_df.iterrows():
    s = row['source_id']
    if s not in sources:
        continue
    try:
        supply_units[s] = float(row['supply_units'])
    except Exception:
        raise ValueError(f"Invalid supply_units for source {s}: {row['supply_units']}")
demand_units = {}
for (_, row) in dest_df.iterrows():
    d = row['destination_id']
    if d not in destinations:
        continue
    try:
        demand_units[d] = float(row['demand_units'])
    except Exception:
        raise ValueError(f"Invalid demand_units for destination {d}: {row['demand_units']}")
TRUCK_CAPACITY = 10.0

def build_model():
    m = gp.Model('TruckTransportation')
    t_vars = m.addVars(sources, destinations, vtype=gp.GRB.INTEGER, lb=0, name='')
    q_vars = m.addVars(sources, destinations, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
    m.setObjective(gp.quicksum((cost[s][d] * q_vars[s, d] for s in sources for d in destinations)), gp.GRB.MINIMIZE)
    for s in sources:
        m.addConstr(gp.quicksum((q_vars[s, d] for d in destinations)) <= supply_units[s], name=f'supply_{s}')
    for d in destinations:
        m.addConstr(gp.quicksum((q_vars[s, d] for s in sources)) == demand_units[d], name=f'demand_{d}')
    for s in sources:
        for d in destinations:
            m.addConstr(q_vars[s, d] <= TRUCK_CAPACITY * t_vars[s, d], name=f'truckload_{s}_{d}')
    return m
m = build_model()
m.optimize()