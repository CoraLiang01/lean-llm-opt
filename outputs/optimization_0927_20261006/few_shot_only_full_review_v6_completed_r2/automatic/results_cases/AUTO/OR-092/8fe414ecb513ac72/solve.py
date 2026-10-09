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
cost_sources = set(cost_df['source_id'].str.strip())
cost_dest_cols = set([c for c in cost_df.columns if c != 'source_id'])
if set(sources) - cost_sources:
    raise ValueError(f'Missing sources in cost matrix: {set(sources) - cost_sources}')
if set(destinations) - cost_dest_cols:
    raise ValueError(f'Missing destinations in cost matrix: {set(destinations) - cost_dest_cols}')
src_ids = set(src_df['source_id'].str.strip())
if set(sources) - src_ids:
    raise ValueError(f'Missing sources in sources file: {set(sources) - src_ids}')
dest_ids = set(dest_df['destination_id'].str.strip())
if set(destinations) - dest_ids:
    raise ValueError(f'Missing destinations in destinations file: {set(destinations) - dest_ids}')
src_df['source_id'] = src_df['source_id'].str.strip()
src_df['supply_units'] = src_df['supply_units'].str.strip()
supply_units = {}
for (_, row) in src_df.iterrows():
    sid = row['source_id']
    if sid in sources:
        try:
            supply_units[sid] = float(row['supply_units'])
        except Exception:
            raise ValueError(f"Invalid supply_units for source {sid}: {row['supply_units']}")
dest_df['destination_id'] = dest_df['destination_id'].str.strip()
dest_df['demand_units'] = dest_df['demand_units'].str.strip()
demand_units = {}
for (_, row) in dest_df.iterrows():
    did = row['destination_id']
    if did in destinations:
        try:
            demand_units[did] = float(row['demand_units'])
        except Exception:
            raise ValueError(f"Invalid demand_units for destination {did}: {row['demand_units']}")
cost_per_unit = {}
for (_, row) in cost_df.iterrows():
    sid = row['source_id'].strip()
    if sid in sources:
        for did in destinations:
            val = row[did].strip()
            try:
                cost_per_unit[sid, did] = float(val)
            except Exception:
                raise ValueError(f'Invalid cost for ({sid},{did}): {val}')
m = gp.Model('TruckTransportation')
t_vars = m.addVars(sources, destinations, vtype=gp.GRB.INTEGER, lb=0, name='')
q_vars = m.addVars(sources, destinations, vtype=gp.GRB.CONTINUOUS, lb=0, name='')
for s in sources:
    m.addConstr(gp.quicksum((q_vars[s, d] for d in destinations)) <= supply_units[s], name=f'supply_{s}')
for d in destinations:
    m.addConstr(gp.quicksum((q_vars[s, d] for s in sources)) == demand_units[d], name=f'demand_{d}')
for s in sources:
    for d in destinations:
        m.addConstr(q_vars[s, d] <= 10 * t_vars[s, d], name=f'truckcap_{s}_{d}')
m.setObjective(gp.quicksum((cost_per_unit[s, d] * q_vars[s, d] for s in sources for d in destinations)), gp.GRB.MINIMIZE)
m.optimize()