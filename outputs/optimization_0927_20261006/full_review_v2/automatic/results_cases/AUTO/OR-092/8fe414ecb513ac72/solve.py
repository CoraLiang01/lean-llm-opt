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

def norm_str(x):
    return x.strip().casefold()
cost_df['source_id_norm'] = cost_df['source_id'].apply(norm_str)
sources_df['source_id_norm'] = sources_df['source_id'].apply(norm_str)
dest_df['destination_id_norm'] = dest_df['destination_id'].apply(norm_str)
S = list(sources_df['source_id'])
S_norm = [norm_str(s) for s in S]
D = list(dest_df['destination_id'])
D_norm = [norm_str(d) for d in D]
supply_units = {}
for (idx, row) in sources_df.iterrows():
    key = norm_str(row['source_id'])
    try:
        supply_units[key] = int(row['supply_units'])
    except Exception:
        raise ValueError(f"Invalid supply_units for source {row['source_id']}: {row['supply_units']}")
demand_units = {}
for (idx, row) in dest_df.iterrows():
    key = norm_str(row['destination_id'])
    try:
        demand_units[key] = int(row['demand_units'])
    except Exception:
        raise ValueError(f"Invalid demand_units for destination {row['destination_id']}: {row['demand_units']}")
cost = {}
cost_columns = [col for col in cost_df.columns if re.fullmatch('D\\d+', col.strip(), re.IGNORECASE)]
if len(cost_columns) != len(D):
    raise ValueError(f'Cost matrix columns {cost_columns} do not match destination list {D}')
for (idx, row) in cost_df.iterrows():
    s_norm = row['source_id_norm']
    for (d, d_norm) in zip(D, D_norm):
        if d not in cost_df.columns:
            raise KeyError(f'Destination {d} not found in cost matrix columns')
        try:
            cost_val = float(row[d])
        except Exception:
            raise ValueError(f"Invalid cost for source {row['source_id']} to destination {d}: {row[d]}")
        cost[s_norm, d_norm] = cost_val
for s in S_norm:
    if s not in supply_units:
        raise KeyError(f'Source {s} missing in supply_units')
    for d in D_norm:
        if (s, d) not in cost:
            raise KeyError(f'Missing cost entry for source {s}, destination {d}')
for d in D_norm:
    if d not in demand_units:
        raise KeyError(f'Destination {d} missing in demand_units')
m = gp.Model('TruckTransportation')
t_vars = m.addVars(S_norm, D_norm, vtype=gp.GRB.INTEGER, lb=0, name='')
x_vars = m.addVars(S_norm, D_norm, vtype=gp.GRB.CONTINUOUS, lb=0, name='')
TRUCK_CAPACITY = 10
for s in S_norm:
    for d in D_norm:
        m.addConstr(x_vars[s, d] <= TRUCK_CAPACITY * t_vars[s, d], name=f'cap_{s}_{d}')
for (s, s_orig) in zip(S_norm, S):
    m.addConstr(gp.quicksum((x_vars[s, d] for d in D_norm)) <= supply_units[s], name=f'supply_{s_orig}')
for (d, d_orig) in zip(D_norm, D):
    m.addConstr(gp.quicksum((x_vars[s, d] for s in S_norm)) == demand_units[d], name=f'demand_{d_orig}')
m.setObjective(gp.quicksum((cost[s, d] * x_vars[s, d] for s in S_norm for d in D_norm)), gp.GRB.MINIMIZE)
m.optimize()