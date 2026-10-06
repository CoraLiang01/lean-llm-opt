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
destinations = [f'D{i}' for i in range(1, 21)]
supply_dict = {}
for _, row in sources_df.iterrows():
    sid = str(row['source_id']).strip()
    if sid in sources:
        supply_dict[sid] = int(row['supply_units'])
if set(sources) != set(supply_dict.keys()):
    missing = set(sources) - set(supply_dict.keys())
    raise ValueError(f'Missing supply data for sources: {missing}')
demand_dict = {}
for _, row in dest_df.iterrows():
    did = str(row['destination_id']).strip()
    if did in destinations:
        demand_dict[did] = int(row['demand_units'])
if set(destinations) != set(demand_dict.keys()):
    missing = set(destinations) - set(demand_dict.keys())
    raise ValueError(f'Missing demand data for destinations: {missing}')
cost_dict = {}
cost_df['source_id'] = cost_df['source_id'].astype(str).str.strip()
for _, row in cost_df.iterrows():
    sid = row['source_id']
    if sid in sources:
        for did in destinations:
            if did not in row:
                raise ValueError(f'Missing cost for ({sid},{did}) in cost matrix')
            cost_dict[sid, did] = float(row[did])
if len(cost_dict) != len(sources) * len(destinations):
    raise ValueError('Cost matrix does not cover all (source, destination) pairs.')
m = gp.Model('TruckTransportation')
t = m.addVars(sources, destinations, vtype=gp.GRB.INTEGER, lb=0, name='')
q = m.addVars(sources, destinations, vtype=gp.GRB.CONTINUOUS, lb=0, name='')
for s in sources:
    m.addConstr(gp.quicksum((q[s, d] for d in destinations)) <= supply_dict[s], name=f'supply_{s}')
for d in destinations:
    m.addConstr(gp.quicksum((q[s, d] for s in sources)) == demand_dict[d], name=f'demand_{d}')
for s in sources:
    for d in destinations:
        m.addConstr(q[s, d] <= 10 * t[s, d], name=f'truckcap_{s}_{d}')
m.setObjective(gp.quicksum((cost_dict[s, d] * q[s, d] for s in sources for d in destinations)), gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total transportation cost: {m.objVal:.2f}')
    print('\nRoute plan (only nonzero truck trips):')
    for s in sources:
        for d in destinations:
            tval = t[s, d].X
            qval = q[s, d].X
            if tval > 1e-06:
                print(f'  {s} -> {d}: {int(round(tval))} trucks, {qval:.2f} units (cost per unit: {cost_dict[s, d]:.2f})')
    print('\nSource utilization:')
    for s in sources:
        total_out = sum((q[s, d].X for d in destinations))
        print(f'  {s}: {total_out:.2f} / {supply_dict[s]} units shipped')
    print('\nDestination fulfillment:')
    for d in destinations:
        total_in = sum((q[s, d].X for s in sources))
        print(f'  {d}: {total_in:.2f} / {demand_dict[d]} units received')
else:
    print(f'No optimal solution found. Status: {m.status}')