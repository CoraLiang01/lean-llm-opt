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
cost_df['source_id'] = cost_df['source_id'].str.strip()
if set(sources) != set(cost_df['source_id']):
    raise ValueError(f"Mismatch in sources: expected {sources}, found {list(cost_df['source_id'])}")
for d in destinations:
    if d not in cost_df.columns:
        raise ValueError(f'Destination {d} missing from cost matrix columns.')
cost = {}
for (_, row) in cost_df.iterrows():
    s = row['source_id']
    cost[s] = {}
    for d in destinations:
        val = row[d]
        try:
            cost[s][d] = float(val)
        except Exception:
            raise ValueError(f'Invalid cost value for source {s}, destination {d}: {val}')
src_df['source_id'] = src_df['source_id'].str.strip()
src_df['supply_units'] = src_df['supply_units'].str.strip()
supply_units = {}
for (_, row) in src_df.iterrows():
    s = row['source_id']
    if s not in sources:
        continue
    try:
        supply_units[s] = float(row['supply_units'])
    except Exception:
        raise ValueError(f"Invalid supply_units for source {s}: {row['supply_units']}")
if set(sources) != set(supply_units.keys()):
    raise ValueError(f'Supply data missing for some sources: {set(sources) - set(supply_units.keys())}')
dest_df['destination_id'] = dest_df['destination_id'].str.strip()
dest_df['demand_units'] = dest_df['demand_units'].str.strip()
demand_units = {}
for (_, row) in dest_df.iterrows():
    d = row['destination_id']
    if d not in destinations:
        continue
    try:
        demand_units[d] = float(row['demand_units'])
    except Exception:
        raise ValueError(f"Invalid demand_units for destination {d}: {row['demand_units']}")
if set(destinations) != set(demand_units.keys()):
    raise ValueError(f'Demand data missing for some destinations: {set(destinations) - set(demand_units.keys())}')
m = gp.Model('TruckTransportation')
t_vars = m.addVars(sources, destinations, vtype=gp.GRB.INTEGER, lb=0, name='')
q_vars = m.addVars(sources, destinations, vtype=gp.GRB.CONTINUOUS, lb=0, name='')
m.setObjective(gp.quicksum((cost[s][d] * q_vars[s, d] for s in sources for d in destinations)), gp.GRB.MINIMIZE)
for s in sources:
    m.addConstr(gp.quicksum((q_vars[s, d] for d in destinations)) <= supply_units[s], name=f'supply_{s}')
for d in destinations:
    m.addConstr(gp.quicksum((q_vars[s, d] for s in sources)) == demand_units[d], name=f'demand_{d}')
for s in sources:
    for d in destinations:
        m.addConstr(q_vars[s, d] <= 10 * t_vars[s, d], name=f'truckcap_{s}_{d}')
m.optimize()