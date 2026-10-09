import gurobipy as gp
import pandas as pd
import numpy as np
import re

def solve_transportation_problem():
    cost_matrix_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_cost_matrix.csv'
    destinations_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_destinations.csv'
    sources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_sources.csv'
    cost_df = pd.read_csv(cost_matrix_path, sep=',', dtype=str, keep_default_na=False)
    dest_df = pd.read_csv(destinations_path, sep=',', dtype=str, keep_default_na=False)
    src_df = pd.read_csv(sources_path, sep=',', dtype=str, keep_default_na=False)
    sources = src_df['source_id'].tolist()
    destinations = dest_df['destination_id'].tolist()
    supply_units = {}
    for (idx, row) in src_df.iterrows():
        sid = row['source_id']
        try:
            supply_units[sid] = float(row['supply_units'])
        except Exception:
            raise ValueError(f"Invalid supply_units for source {sid}: {row['supply_units']}")
    demand_units = {}
    for (idx, row) in dest_df.iterrows():
        did = row['destination_id']
        try:
            demand_units[did] = float(row['demand_units'])
        except Exception:
            raise ValueError(f"Invalid demand_units for destination {did}: {row['demand_units']}")
    cost_sources = cost_df['source_id'].tolist()
    if set(sources) != set(cost_sources):
        raise ValueError(f'Mismatch between sources in expanded_sources.csv and expanded_cost_matrix.csv: {set(sources) ^ set(cost_sources)}')
    cost_dest_cols = [col for col in cost_df.columns if col != 'source_id']
    if set(destinations) != set(cost_dest_cols):
        raise ValueError(f'Mismatch between destinations in expanded_destinations.csv and expanded_cost_matrix.csv: {set(destinations) ^ set(cost_dest_cols)}')
    cost = {}
    for (idx, row) in cost_df.iterrows():
        sid = row['source_id']
        for did in destinations:
            try:
                cost_val = float(row[did])
            except Exception:
                raise ValueError(f'Invalid cost for source {sid}, destination {did}: {row[did]}')
            cost[sid, did] = cost_val
    route_keys = [(s, d) for s in sources for d in destinations]
    m = gp.Model('Transportation_MILP')
    t_vars = m.addVars(route_keys, vtype=gp.GRB.INTEGER, lb=0, name='')
    q_vars = m.addVars(route_keys, vtype=gp.GRB.CONTINUOUS, lb=0, name='')
    m.setObjective(gp.quicksum((cost[s, d] * q_vars[s, d] for (s, d) in route_keys)), gp.GRB.MINIMIZE)
    for s in sources:
        m.addConstr(gp.quicksum((q_vars[s, d] for d in destinations)) <= supply_units[s], name=f'supply_{s}')
    for d in destinations:
        m.addConstr(gp.quicksum((q_vars[s, d] for s in sources)) == demand_units[d], name=f'demand_{d}')
    truck_capacity = 10.0
    for (s, d) in route_keys:
        m.addConstr(q_vars[s, d] <= truck_capacity * t_vars[s, d], name=f'truckcap_{s}_{d}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        print(f'ObjVal: {m.objVal}')
        for (s, d) in route_keys:
            print(f'{q_vars[s, d].VarName} {q_vars[s, d].X}')
            print(f'{t_vars[s, d].VarName} {t_vars[s, d].X}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_transportation_problem()