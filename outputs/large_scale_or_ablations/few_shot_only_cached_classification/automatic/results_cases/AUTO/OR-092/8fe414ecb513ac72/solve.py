import gurobipy as gp
import pandas as pd
import numpy as np
cost_matrix_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_cost_matrix.csv'
sources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_sources.csv'
destinations_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_destinations.csv'
cost_df = pd.read_csv(cost_matrix_path, sep=',')
sources_df = pd.read_csv(sources_path, sep=',')
dest_df = pd.read_csv(destinations_path, sep=',')
sources = [f'S{i}' for i in range(1, 11)]
destinations = [f'D{j}' for j in range(1, 21)]
missing_sources = set(sources) - set(sources_df['source_id'].astype(str))
if missing_sources:
    raise ValueError(f'Missing sources in sources_df: {missing_sources}')
missing_destinations = set(destinations) - set(dest_df['destination_id'].astype(str))
if missing_destinations:
    raise ValueError(f'Missing destinations in dest_df: {missing_destinations}')
supply = dict(zip(sources_df['source_id'].astype(str), sources_df['supply_units']))
demand = dict(zip(dest_df['destination_id'].astype(str), dest_df['demand_units']))
cost = {}
cost_df = cost_df.set_index('source_id')
for i in sources:
    if i not in cost_df.index:
        raise ValueError(f'Source {i} missing in cost matrix')
    for j in destinations:
        if j not in cost_df.columns:
            raise ValueError(f'Destination {j} missing in cost matrix columns')
        cost[i, j] = float(cost_df.loc[i, j])
m = gp.Model('TruckTransportation')
t = m.addVars(sources, destinations, vtype=gp.GRB.INTEGER, lb=0, name='')
f = m.addVars(sources, destinations, vtype=gp.GRB.CONTINUOUS, lb=0, name='')
m.setObjective(gp.quicksum((cost[i, j] * f[i, j] for i in sources for j in destinations)), gp.GRB.MINIMIZE)
for i in sources:
    m.addConstr(gp.quicksum((f[i, j] for j in destinations)) <= supply[i], name=f'supply_{i}')
for j in destinations:
    m.addConstr(gp.quicksum((f[i, j] for i in sources)) == demand[j], name=f'demand_{j}')
for i in sources:
    for j in destinations:
        m.addConstr(f[i, j] <= 10 * t[i, j], name=f'truckcap_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total transportation cost: {m.objVal:.2f}')
    print('\nRoute plan (only nonzero shipments):')
    for i in sources:
        for j in destinations:
            shipped = f[i, j].X
            trucks = t[i, j].X
            if shipped > 1e-06:
                print(f'  {i} -> {j}: {shipped:.2f} units in {int(round(trucks))} truck(s)')
else:
    print(f'No optimal solution found. Status: {m.status}')