import gurobipy as gp
import pandas as pd
import numpy as np
import re
cost_matrix_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_cost_matrix.csv'
destinations_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_destinations.csv'
sources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_sources.csv'
cost_df = pd.read_csv(cost_matrix_path, sep=',', dtype=str, keep_default_na=False)
dest_df = pd.read_csv(destinations_path, sep=',', dtype=str, keep_default_na=False)
src_df = pd.read_csv(sources_path, sep=',', dtype=str, keep_default_na=False)
sources = src_df['source_id'].astype(str).str.strip().tolist()
destinations = dest_df['destination_id'].astype(str).str.strip().tolist()
cost_sources = cost_df['source_id'].astype(str).str.strip().tolist()
cost_destinations = [c for c in cost_df.columns if c != 'source_id']
if set(sources) != set(cost_sources):
    raise ValueError(f'Mismatch between sources in expanded_sources.csv and expanded_cost_matrix.csv: {set(sources)} vs {set(cost_sources)}')
if set(destinations) != set(cost_destinations):
    raise ValueError(f'Mismatch between destinations in expanded_destinations.csv and expanded_cost_matrix.csv: {set(destinations)} vs {set(cost_destinations)}')
cost = {}
for _, row in cost_df.iterrows():
    s = str(row['source_id']).strip()
    for d in destinations:
        val = row[d]
        try:
            cost[s, d] = float(val)
        except Exception:
            raise ValueError(f'Invalid cost value for ({s},{d}): {val}')
supply = {}
for _, row in src_df.iterrows():
    s = str(row['source_id']).strip()
    try:
        supply[s] = int(row['supply_units'])
    except Exception:
        raise ValueError(f"Invalid supply_units for {s}: {row['supply_units']}")
demand = {}
for _, row in dest_df.iterrows():
    d = str(row['destination_id']).strip()
    try:
        demand[d] = int(row['demand_units'])
    except Exception:
        raise ValueError(f"Invalid demand_units for {d}: {row['demand_units']}")
m = gp.Model('TruckTransportation')
t_vars = m.addVars(sources, destinations, vtype=gp.GRB.INTEGER, lb=0, name='')
x_vars = m.addVars(sources, destinations, vtype=gp.GRB.CONTINUOUS, lb=0, name='')
m.setObjective(gp.quicksum((cost[s, d] * x_vars[s, d] for s in sources for d in destinations)), gp.GRB.MINIMIZE)
for s in sources:
    m.addConstr(gp.quicksum((x_vars[s, d] for d in destinations)) <= supply[s], name='')
for d in destinations:
    m.addConstr(gp.quicksum((x_vars[s, d] for s in sources)) == demand[d], name='')
for s in sources:
    for d in destinations:
        m.addConstr(x_vars[s, d] <= 10 * t_vars[s, d], name='')
m.optimize()