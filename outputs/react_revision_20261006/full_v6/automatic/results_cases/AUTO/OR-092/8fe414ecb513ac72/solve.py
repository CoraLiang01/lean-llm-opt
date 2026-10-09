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
missing_sources = set(sources) - set(cost_df['source_id'])
if missing_sources:
    raise ValueError(f'Missing sources in cost matrix: {missing_sources}')
missing_sources2 = set(sources) - set(sources_df['source_id'])
if missing_sources2:
    raise ValueError(f'Missing sources in sources file: {missing_sources2}')
missing_dest = set(destinations) - set(dest_df['destination_id'])
if missing_dest:
    raise ValueError(f'Missing destinations in destinations file: {missing_dest}')
missing_dest2 = set(destinations) - set(cost_df.columns[1:])
if missing_dest2:
    raise ValueError(f'Missing destinations in cost matrix columns: {missing_dest2}')
cost = {}
for (_, row) in cost_df.iterrows():
    s = row['source_id']
    for d in destinations:
        try:
            cost_val = float(row[d])
        except Exception:
            raise ValueError(f'Missing or invalid cost for route ({s}, {d})')
        cost[s, d] = cost_val
supply = {}
for (_, row) in sources_df.iterrows():
    s = row['source_id']
    try:
        supply_val = int(row['supply_units'])
    except Exception:
        raise ValueError(f'Missing or invalid supply_units for source {s}')
    supply[s] = supply_val
demand = {}
for (_, row) in dest_df.iterrows():
    d = row['destination_id']
    try:
        demand_val = int(row['demand_units'])
    except Exception:
        raise ValueError(f'Missing or invalid demand_units for destination {d}')
    demand[d] = demand_val
for s in sources:
    for d in destinations:
        if (s, d) not in cost:
            raise ValueError(f'Missing cost for ({s}, {d})')
TRUCK_CAPACITY = 10

def solve_transportation():
    m = gp.Model('TruckTransportation')
    m.Params.MIPGap = 0.0001
    t_keys = [(s, d) for s in sources for d in destinations]
    t_vars = m.addVars(t_keys, vtype=gp.GRB.INTEGER, lb=0, name='')
    x_vars = m.addVars(t_keys, vtype=gp.GRB.CONTINUOUS, lb=0, name='')
    m.setObjective(gp.quicksum((cost[s, d] * x_vars[s, d] for s in sources for d in destinations)), gp.GRB.MINIMIZE)
    for s in sources:
        m.addConstr(gp.quicksum((x_vars[s, d] for d in destinations)) <= supply[s], name=f'supply_{s}')
    for d in destinations:
        m.addConstr(gp.quicksum((x_vars[s, d] for s in sources)) == demand[d], name=f'demand_{d}')
    for s in sources:
        for d in destinations:
            m.addConstr(x_vars[s, d] <= TRUCK_CAPACITY * t_vars[s, d], name=f'truckcap_{s}_{d}')
    m.optimize()
    return m
m = solve_transportation()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')