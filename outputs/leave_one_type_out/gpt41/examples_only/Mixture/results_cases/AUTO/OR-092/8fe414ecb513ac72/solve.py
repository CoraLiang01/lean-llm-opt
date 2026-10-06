import gurobipy as gp
import pandas as pd
import numpy as np
import re
cost_matrix_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_cost_matrix.csv'
sources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_sources.csv'
destinations_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_destinations.csv'
cost_df = pd.read_csv(cost_matrix_path, sep=',')
sources_df = pd.read_csv(sources_path, sep=',')
dest_df = pd.read_csv(destinations_path, sep=',')
sources = sources_df['source_id'].astype(str).tolist()
if set(sources) != set(cost_df['source_id'].astype(str)):
    raise ValueError('Mismatch between sources in sources.csv and cost_matrix.csv')
destinations = dest_df['destination_id'].astype(str).tolist()
cost_dest_cols = [col for col in cost_df.columns if col != 'source_id']
if set(destinations) != set(cost_dest_cols):
    raise ValueError('Mismatch between destinations in destinations.csv and cost_matrix.csv')
supply = dict(zip(sources_df['source_id'].astype(str), sources_df['supply_units']))
demand = dict(zip(dest_df['destination_id'].astype(str), dest_df['demand_units']))
cost = {}
for _, row in cost_df.iterrows():
    s = str(row['source_id'])
    for d in destinations:
        cost[s, d] = float(row[d])
TRUCK_CAPACITY = 10
m = gp.Model('Transportation_MILP')
t = m.addVars(sources, destinations, vtype=gp.GRB.INTEGER, lb=0, name='')
q = m.addVars(sources, destinations, vtype=gp.GRB.CONTINUOUS, lb=0, name='')
m.setObjective(gp.quicksum((cost[s, d] * q[s, d] for s in sources for d in destinations)), gp.GRB.MINIMIZE)
for s in sources:
    m.addConstr(gp.quicksum((q[s, d] for d in destinations)) <= supply[s], name=f'supply_{s}')
for d in destinations:
    m.addConstr(gp.quicksum((q[s, d] for s in sources)) == demand[d], name=f'demand_{d}')
for s in sources:
    for d in destinations:
        m.addConstr(q[s, d] <= TRUCK_CAPACITY * t[s, d], name=f'truckcap_{s}_{d}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total transportation cost: {m.objVal:.2f}')
    print('\nRoute plan (only nonzero shipments):')
    for s in sources:
        for d in destinations:
            qval = q[s, d].X
            tval = t[s, d].X
            if qval > 1e-06:
                print(f'  {s} -> {d}: {qval:.2f} units in {int(round(tval))} truck(s)')
else:
    print(f'No optimal solution found. Status: {m.status}')