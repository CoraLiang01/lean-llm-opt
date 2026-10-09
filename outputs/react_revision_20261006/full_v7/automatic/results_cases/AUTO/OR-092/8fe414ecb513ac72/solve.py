import gurobipy as gp
import pandas as pd
import numpy as np
cost_matrix_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_cost_matrix.csv'
sources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_sources.csv'
destinations_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_destinations.csv'
cost_df = pd.read_csv(cost_matrix_path, dtype=str, keep_default_na=False)
sources_df = pd.read_csv(sources_path, dtype=str, keep_default_na=False)
dest_df = pd.read_csv(destinations_path, dtype=str, keep_default_na=False)
cost_df['source_id'] = cost_df['source_id'].str.strip()
sources_df['source_id'] = sources_df['source_id'].str.strip()
dest_df['destination_id'] = dest_df['destination_id'].str.strip()
sources = [f'S{i}' for i in range(1, 11)]
destinations = [f'D{i}' for i in range(1, 21)]
missing_sources = set(sources) - set(sources_df['source_id'])
if missing_sources:
    raise ValueError(f'Missing sources in expanded_sources.csv: {missing_sources}')
missing_dest = set(destinations) - set(dest_df['destination_id'])
if missing_dest:
    raise ValueError(f'Missing destinations in expanded_destinations.csv: {missing_dest}')
missing_cost_sources = set(sources) - set(cost_df['source_id'])
if missing_cost_sources:
    raise ValueError(f'Missing sources in expanded_cost_matrix.csv: {missing_cost_sources}')
missing_cost_dest = set(destinations) - set(cost_df.columns[1:])
if missing_cost_dest:
    raise ValueError(f'Missing destinations in expanded_cost_matrix.csv columns: {missing_cost_dest}')
supply_units = {}
for (_, row) in sources_df.iterrows():
    sid = row['source_id']
    if sid in sources:
        try:
            supply_units[sid] = int(row['supply_units'])
        except Exception:
            raise ValueError(f"Non-integer supply_units for source {sid}: {row['supply_units']}")
demand_units = {}
for (_, row) in dest_df.iterrows():
    did = row['destination_id']
    if did in destinations:
        try:
            demand_units[did] = int(row['demand_units'])
        except Exception:
            raise ValueError(f"Non-integer demand_units for destination {did}: {row['demand_units']}")
cost = {}
cost_df_indexed = cost_df.set_index('source_id')
for s in sources:
    if s not in cost_df_indexed.index:
        raise ValueError(f'Source {s} missing in cost matrix')
    for d in destinations:
        try:
            val = cost_df_indexed.loc[s, d]
            cost[s, d] = float(val)
        except Exception:
            raise ValueError(f'Missing or non-numeric cost for ({s},{d}): {val}')
sd_pairs = [(s, d) for s in sources for d in destinations]

def solve_transportation():
    m = gp.Model('TruckTransportation')
    m.Params.MIPGap = 0.0001
    t_vars = m.addVars(sd_pairs, vtype=gp.GRB.INTEGER, lb=0, name='')
    x_vars = m.addVars(sd_pairs, vtype=gp.GRB.CONTINUOUS, lb=0, name='')
    m.setObjective(gp.quicksum((cost[s, d] * x_vars[s, d] for (s, d) in sd_pairs)), gp.GRB.MINIMIZE)
    for s in sources:
        m.addConstr(gp.quicksum((x_vars[s, d] for d in destinations)) <= supply_units[s], name=f'supply_{s}')
    for d in destinations:
        m.addConstr(gp.quicksum((x_vars[s, d] for s in sources)) == demand_units[d], name=f'demand_{d}')
    for (s, d) in sd_pairs:
        m.addConstr(x_vars[s, d] <= 10 * t_vars[s, d], name=f'truckcap_{s}_{d}')
    m.optimize()
    return m
m = solve_transportation()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')