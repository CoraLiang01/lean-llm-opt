import gurobipy as gp
import pandas as pd
import numpy as np
cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_cost_matrix.csv', sep=',')
sources_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_sources.csv', sep=',')
dest_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_destinations.csv', sep=',')
sources = ['S%d' % i for i in range(1, 11)]
destinations = ['D%d' % i for i in range(1, 21)]
if not set(sources).issubset(set(sources_df['source_id'].astype(str))):
    missing = set(sources) - set(sources_df['source_id'].astype(str))
    raise ValueError(f'Missing sources in expanded_sources.csv: {missing}')
if not set(destinations).issubset(set(dest_df['destination_id'].astype(str))):
    missing = set(destinations) - set(dest_df['destination_id'].astype(str))
    raise ValueError(f'Missing destinations in expanded_destinations.csv: {missing}')
if not set(sources).issubset(set(cost_df['source_id'].astype(str))):
    missing = set(sources) - set(cost_df['source_id'].astype(str))
    raise ValueError(f'Missing sources in expanded_cost_matrix.csv: {missing}')
if not set(destinations).issubset(set(cost_df.columns[1:])):
    missing = set(destinations) - set(cost_df.columns[1:])
    raise ValueError(f'Missing destinations in expanded_cost_matrix.csv: {missing}')
supply = dict(zip(sources_df['source_id'].astype(str), sources_df['supply_units']))
demand = dict(zip(dest_df['destination_id'].astype(str), dest_df['demand_units']))
cost = {}
for _, row in cost_df.iterrows():
    i = str(row['source_id'])
    for j in destinations:
        cost[i, j] = float(row[j])
truck_capacity = 10
m = gp.Model('Transportation_MILP')
t = m.addVars(sources, destinations, vtype=gp.GRB.INTEGER, lb=0, name='')
q = m.addVars(sources, destinations, vtype=gp.GRB.CONTINUOUS, lb=0, name='')
m.setObjective(gp.quicksum((cost[i, j] * q[i, j] for i in sources for j in destinations)), gp.GRB.MINIMIZE)
for i in sources:
    m.addConstr(gp.quicksum((q[i, j] for j in destinations)) <= supply[i], name=f'supply_{i}')
for j in destinations:
    m.addConstr(gp.quicksum((q[i, j] for i in sources)) == demand[j], name=f'demand_{j}')
for i in sources:
    for j in destinations:
        m.addConstr(q[i, j] <= truck_capacity * t[i, j], name=f'truckcap_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total transportation cost: {m.objVal:.2f}')
    print('\nRoute Plan (only nonzero shipments):')
    for i in sources:
        for j in destinations:
            qval = q[i, j].X
            tval = t[i, j].X
            if qval > 1e-06:
                print(f'  {i} -> {j}: {qval:.2f} units in {int(round(tval))} truck(s)')
else:
    print(f'No optimal solution found. Status: {m.status}')