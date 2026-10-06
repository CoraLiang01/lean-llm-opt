import gurobipy as gp
import pandas as pd
import numpy as np
cost_matrix_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_cost_matrix.csv'
sources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_sources.csv'
destinations_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_destinations.csv'
cost_df = pd.read_csv(cost_matrix_path, sep=',')
sources_df = pd.read_csv(sources_path, sep=',')
dest_df = pd.read_csv(destinations_path, sep=',')
cost_df['source_id'] = cost_df['source_id'].astype(str).str.strip()
sources_df['source_id'] = sources_df['source_id'].astype(str).str.strip()
dest_df['destination_id'] = dest_df['destination_id'].astype(str).str.strip()
sources = [f'S{i}' for i in range(1, 11)]
destinations = [f'D{j}' for j in range(1, 21)]
missing_sources = set(sources) - set(cost_df['source_id'])
if missing_sources:
    raise ValueError(f'Missing sources in cost matrix: {missing_sources}')
missing_sources2 = set(sources) - set(sources_df['source_id'])
if missing_sources2:
    raise ValueError(f'Missing sources in sources file: {missing_sources2}')
missing_dest = set(destinations) - set(cost_df.columns[1:])
if missing_dest:
    raise ValueError(f'Missing destinations in cost matrix columns: {missing_dest}')
missing_dest2 = set(destinations) - set(dest_df['destination_id'])
if missing_dest2:
    raise ValueError(f'Missing destinations in destinations file: {missing_dest2}')
cost = {}
for i in sources:
    row = cost_df.loc[cost_df['source_id'] == i]
    if row.empty:
        raise ValueError(f'Source {i} not found in cost matrix.')
    for j in destinations:
        val = float(row.iloc[0][j])
        cost[i, j] = val
supply = {}
for i in sources:
    row = sources_df.loc[sources_df['source_id'] == i]
    if row.empty:
        raise ValueError(f'Source {i} not found in sources file.')
    supply[i] = int(row.iloc[0]['supply_units'])
demand = {}
for j in destinations:
    row = dest_df.loc[dest_df['destination_id'] == j]
    if row.empty:
        raise ValueError(f'Destination {j} not found in destinations file.')
    demand[j] = int(row.iloc[0]['demand_units'])
m = gp.Model('TruckTransportation')
t = m.addVars(sources, destinations, vtype=gp.GRB.INTEGER, lb=0, name='')
x = m.addVars(sources, destinations, vtype=gp.GRB.CONTINUOUS, lb=0, name='')
m.setObjective(gp.quicksum((cost[i, j] * x[i, j] for i in sources for j in destinations)), gp.GRB.MINIMIZE)
for i in sources:
    m.addConstr(gp.quicksum((x[i, j] for j in destinations)) <= supply[i], name=f'supply_{i}')
for j in destinations:
    m.addConstr(gp.quicksum((x[i, j] for i in sources)) == demand[j], name=f'demand_{j}')
for i in sources:
    for j in destinations:
        m.addConstr(x[i, j] <= 10 * t[i, j], name=f'truckcap_{i}_{j}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal {m.objVal}')
    for i in sources:
        for j in destinations:
            print(f'x[{i},{j}] {x[i, j].VarName} {x[i, j].X}')
            print(f't[{i},{j}] {t[i, j].VarName} {t[i, j].X}')
else:
    print(f'Solver status: {m.status}')