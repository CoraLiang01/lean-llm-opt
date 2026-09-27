import gurobipy as gp
import pandas as pd
import numpy as np
cost_matrix_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_cost_matrix.csv'
destinations_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_destinations.csv'
sources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_sources.csv'
cost_df = pd.read_csv(cost_matrix_path, sep=',')
dest_df = pd.read_csv(destinations_path, sep=',')
src_df = pd.read_csv(sources_path, sep=',')
sources = [f'S{i}' for i in range(1, 11)]
destinations = [f'D{j}' for j in range(1, 21)]
missing_sources = set(sources) - set(src_df['source_id'])
if missing_sources:
    raise ValueError(f'Missing sources in expanded_sources.csv: {missing_sources}')
missing_destinations = set(destinations) - set(dest_df['destination_id'])
if missing_destinations:
    raise ValueError(f'Missing destinations in expanded_destinations.csv: {missing_destinations}')
supply_dict = dict(zip(src_df['source_id'].astype(str), src_df['supply_units']))
demand_dict = dict(zip(dest_df['destination_id'].astype(str), dest_df['demand_units']))
cost_dict = {}
cost_df = cost_df.set_index('source_id')
for s in sources:
    if s not in cost_df.index:
        raise ValueError(f'Source {s} missing in expanded_cost_matrix.csv')
    for d in destinations:
        if d not in cost_df.columns:
            raise ValueError(f'Destination {d} missing in expanded_cost_matrix.csv')
        cost_dict[s, d] = float(cost_df.loc[s, d])
m = gp.Model('TruckTransportation')
t = m.addVars(sources, destinations, vtype=gp.GRB.INTEGER, lb=0, name='')
x = m.addVars(sources, destinations, vtype=gp.GRB.CONTINUOUS, lb=0, name='')
m.setObjective(gp.quicksum((cost_dict[s, d] * x[s, d] for s in sources for d in destinations)), gp.GRB.MINIMIZE)
for s in sources:
    for d in destinations:
        m.addConstr(x[s, d] <= 10 * t[s, d], name=f'truck_load_{s}_{d}')
for s in sources:
    m.addConstr(gp.quicksum((x[s, d] for d in destinations)) <= supply_dict[s], name=f'supply_{s}')
for d in destinations:
    m.addConstr(gp.quicksum((x[s, d] for s in sources)) == demand_dict[d], name=f'demand_{d}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total transportation cost: {m.objVal:.2f}')
    print('\nRoute plan (only nonzero shipments):')
    for s in sources:
        for d in destinations:
            xval = x[s, d].X
            tval = t[s, d].X
            if xval > 1e-06:
                print(f'  {s} -> {d}: {xval:.2f} units in {int(round(tval))} truck(s) (cost per unit: {cost_dict[s, d]})')
    print('\nSource utilization:')
    for s in sources:
        total_out = sum((x[s, d].X for d in destinations))
        print(f'  {s}: {total_out:.2f} / {supply_dict[s]} units shipped')
    print('\nDestination fulfillment:')
    for d in destinations:
        total_in = sum((x[s, d].X for s in sources))
        print(f'  {d}: {total_in:.2f} / {demand_dict[d]} units received')
else:
    print(f'No optimal solution found. Status: {m.status}')