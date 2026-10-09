import gurobipy as gp
import pandas as pd
import numpy as np

def solve_problem():
    cost_matrix_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_cost_matrix.csv'
    destinations_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_destinations.csv'
    sources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_sources.csv'
    cost_df = pd.read_csv(cost_matrix_path, sep=',', dtype=str, keep_default_na=False)
    dest_df = pd.read_csv(destinations_path, sep=',', dtype=str, keep_default_na=False)
    src_df = pd.read_csv(sources_path, sep=',', dtype=str, keep_default_na=False)
    sources = [f'S{i}' for i in range(1, 11)]
    destinations = [f'D{i}' for i in range(1, 21)]
    cost_sources = set(cost_df['source_id'].tolist())
    if set(sources) - cost_sources:
        raise ValueError(f'Missing sources in cost matrix: {set(sources) - cost_sources}')
    cost_dest_cols = set(cost_df.columns) - {'source_id'}
    if set(destinations) - cost_dest_cols:
        raise ValueError(f'Missing destinations in cost matrix: {set(destinations) - cost_dest_cols}')
    src_ids = set(src_df['source_id'].tolist())
    if set(sources) - src_ids:
        raise ValueError(f'Missing sources in sources.csv: {set(sources) - src_ids}')
    dest_ids = set(dest_df['destination_id'].tolist())
    if set(destinations) - dest_ids:
        raise ValueError(f'Missing destinations in destinations.csv: {set(destinations) - dest_ids}')
    supply_units = {}
    for (_, row) in src_df.iterrows():
        sid = row['source_id']
        if sid in sources:
            try:
                supply_units[sid] = float(row['supply_units'])
            except Exception:
                raise ValueError(f"Non-numeric supply_units for source {sid}: {row['supply_units']}")
    demand_units = {}
    for (_, row) in dest_df.iterrows():
        did = row['destination_id']
        if did in destinations:
            try:
                demand_units[did] = float(row['demand_units'])
            except Exception:
                raise ValueError(f"Non-numeric demand_units for destination {did}: {row['demand_units']}")
    cost = {}
    cost_df_indexed = cost_df.set_index('source_id')
    for s in sources:
        for d in destinations:
            try:
                val = cost_df_indexed.loc[s, d]
                cost[s, d] = float(val)
            except Exception:
                raise ValueError(f"Missing or non-numeric cost for ({s},{d}): {(val if 'val' in locals() else 'MISSING')}")
    m = gp.Model('Truck_Dispatch')
    t_keys = [(s, d) for s in sources for d in destinations]
    t_vars = m.addVars(t_keys, vtype=gp.GRB.INTEGER, lb=0, name='')
    x_vars = m.addVars(t_keys, vtype=gp.GRB.CONTINUOUS, lb=0, name='')
    for s in sources:
        m.addConstr(gp.quicksum((x_vars[s, d] for d in destinations)) <= supply_units[s], name=f'supply_{s}')
    for d in destinations:
        m.addConstr(gp.quicksum((x_vars[s, d] for s in sources)) == demand_units[d], name=f'demand_{d}')
    for (s, d) in t_keys:
        m.addConstr(x_vars[s, d] <= 10 * t_vars[s, d], name=f'truckload_{s}_{d}')
    m.setObjective(gp.quicksum((cost[s, d] * x_vars[s, d] for (s, d) in t_keys)), gp.GRB.MINIMIZE)
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        print(f'ObjVal: {m.objVal}')
        for (s, d) in t_keys:
            print(f'{x_vars[s, d].VarName} {x_vars[s, d].X}')
        for (s, d) in t_keys:
            print(f'{t_vars[s, d].VarName} {t_vars[s, d].X}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_problem()