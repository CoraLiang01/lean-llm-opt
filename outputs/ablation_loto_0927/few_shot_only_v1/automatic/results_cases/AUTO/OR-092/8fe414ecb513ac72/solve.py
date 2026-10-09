import gurobipy as gp
import pandas as pd
import numpy as np
import re
cost_matrix_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_cost_matrix.csv'
destinations_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_destinations.csv'
sources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_sources.csv'
cost_df = pd.read_csv(cost_matrix_path, sep=',')
dest_df = pd.read_csv(destinations_path, sep=',')
source_df = pd.read_csv(sources_path, sep=',')
sources = [f'S{i}' for i in range(1, 11)]
destinations = [f'D{j}' for j in range(1, 21)]
if not set(sources).issubset(set(cost_df['source_id'])):
    missing = set(sources) - set(cost_df['source_id'])
    raise ValueError(f'Missing sources in cost matrix: {missing}')
if not set(sources).issubset(set(source_df['source_id'])):
    missing = set(sources) - set(source_df['source_id'])
    raise ValueError(f'Missing sources in sources file: {missing}')
if not set(destinations).issubset(set(cost_df.columns)):
    missing = set(destinations) - set(cost_df.columns)
    raise ValueError(f'Missing destinations in cost matrix columns: {missing}')
if not set(destinations).issubset(set(dest_df['destination_id'])):
    missing = set(destinations) - set(dest_df['destination_id'])
    raise ValueError(f'Missing destinations in destinations file: {missing}')
cost = {}
for (_, row) in cost_df.iterrows():
    i = str(row['source_id'])
    cost[i] = {}
    for j in destinations:
        cost[i][j] = float(row[j])
supply = {}
for (_, row) in source_df.iterrows():
    i = str(row['source_id'])
    supply[i] = float(row['supply_units'])
demand = {}
for (_, row) in dest_df.iterrows():
    j = str(row['destination_id'])
    demand[j] = float(row['demand_units'])
m = gp.Model('IntegerTruckTransportation')
t = m.addVars(sources, destinations, vtype=gp.GRB.INTEGER, lb=0, name='')
x = m.addVars(sources, destinations, vtype=gp.GRB.CONTINUOUS, lb=0, name='')
m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in sources for j in destinations)), gp.GRB.MINIMIZE)
for i in sources:
    m.addConstr(gp.quicksum((x[i, j] for j in destinations)) <= supply[i], name=f'supply_{i}')
for j in destinations:
    m.addConstr(gp.quicksum((x[i, j] for i in sources)) == demand[j], name=f'demand_{j}')
for i in sources:
    for j in destinations:
        m.addConstr(x[i, j] <= 10 * t[i, j], name=f'truckcap_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total transportation cost: {m.objVal:.2f}')
    print('\nRoute plan (only nonzero shipments):')
    for i in sources:
        for j in destinations:
            xval = x[i, j].X
            tval = t[i, j].X
            if xval > 1e-06:
                print(f'  {i} -> {j}: {xval:.2f} units in {int(round(tval))} truck(s)')
else:
    print(f'No optimal solution found. Status: {m.status}')