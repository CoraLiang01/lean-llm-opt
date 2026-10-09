import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
cost_matrix_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_cost_matrix.csv'
destinations_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_destinations.csv'
sources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_sources.csv'
cost_df = pd.read_csv(cost_matrix_path, dtype=str, keep_default_na=False)
dest_df = pd.read_csv(destinations_path, dtype=str, keep_default_na=False)
src_df = pd.read_csv(sources_path, dtype=str, keep_default_na=False)
cost_df['source_id'] = cost_df['source_id'].str.strip()
src_df['source_id'] = src_df['source_id'].str.strip()
dest_df['destination_id'] = dest_df['destination_id'].str.strip()
sources = [f'S{i}' for i in range(1, 11)]
destinations = [f'D{i}' for i in range(1, 21)]
missing_sources = set(sources) - set(src_df['source_id'])
if missing_sources:
    raise ValueError(f'Missing sources in expanded_sources.csv: {missing_sources}')
missing_destinations = set(destinations) - set(dest_df['destination_id'])
if missing_destinations:
    raise ValueError(f'Missing destinations in expanded_destinations.csv: {missing_destinations}')
src_df['supply_units'] = src_df['supply_units'].astype(int)
supply = {row['source_id']: row['supply_units'] for (_, row) in src_df.iterrows() if row['source_id'] in sources}
dest_df['demand_units'] = dest_df['demand_units'].astype(int)
demand = {row['destination_id']: row['demand_units'] for (_, row) in dest_df.iterrows() if row['destination_id'] in destinations}
cost_dict = {}
cost_df_indexed = cost_df.set_index('source_id')
for s in sources:
    if s not in cost_df_indexed.index:
        raise ValueError(f'Source {s} missing in expanded_cost_matrix.csv')
    for d in destinations:
        if d not in cost_df_indexed.columns:
            raise ValueError(f'Destination {d} missing in expanded_cost_matrix.csv columns')
        val = cost_df_indexed.loc[s, d]
        try:
            cost = float(val)
        except Exception:
            raise ValueError(f'Invalid cost value for ({s},{d}): {val}')
        cost_dict[s, d] = cost
sd_pairs = [(s, d) for s in sources for d in destinations]
m = gp.Model('transportation_milp')
cargo_vars = m.addVars(sd_pairs, lb=0.0, vtype=GRB.CONTINUOUS, name='')
trucks_vars = m.addVars(sd_pairs, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((cost_dict[s, d] * cargo_vars[s, d] for (s, d) in sd_pairs)), GRB.MINIMIZE)
for s in sources:
    m.addConstr(gp.quicksum((cargo_vars[s, d] for d in destinations)) <= supply[s], name=f'supply_{s}')
for d in destinations:
    m.addConstr(gp.quicksum((cargo_vars[s, d] for s in sources)) == demand[d], name=f'demand_{d}')
TRUCK_CAPACITY = 10
for (s, d) in sd_pairs:
    m.addConstr(cargo_vars[s, d] <= TRUCK_CAPACITY * trucks_vars[s, d], name=f'truckload_{s}_{d}')
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for (s, d) in sd_pairs:
        print(f'cargo[{s},{d}] {cargo_vars[s, d].X}')
        print(f'trucks[{s},{d}] {trucks_vars[s, d].X}')
else:
    print(f'Solver status: {m.Status}')