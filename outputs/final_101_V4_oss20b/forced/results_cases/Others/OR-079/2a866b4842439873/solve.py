import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
fac_df = pd.read_csv('/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/facility_costs.csv')
ship_df = pd.read_csv('/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/shipping_costs.csv')
dem_df = pd.read_csv('/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/demand_requirements.csv')
facilities = fac_df['Facility'].tolist()
destinations = dem_df['Destination'].tolist()
fixed_cost = dict(zip(fac_df['Facility'], fac_df['FixedCost']))
capacity = dict(zip(fac_df['Facility'], fac_df['Capacity']))
demand = dict(zip(dem_df['Destination'], dem_df['Demand']))
ship_df_indexed = ship_df.set_index('Origin')
shipping_cost = {fac: {dest: ship_df_indexed.loc[fac, dest] for dest in destinations} for fac in facilities}
m = gp.Model('CFLP')
x = m.addVars(facilities, destinations)
y = m.addVars(facilities, vtype=gp.GRB.BINARY)
obj = gp.quicksum((fixed_cost[fac] * y[fac] for fac in facilities)) + gp.quicksum((shipping_cost[fac][dest] * x[fac, dest] for fac in facilities for dest in destinations))
m.setObjective(obj, gp.GRB.MINIMIZE)
for dest in destinations:
    m.addConstr(gp.quicksum((x[fac, dest] for fac in facilities)) == demand[dest])
for fac in facilities:
    m.addConstr(gp.quicksum((x[fac, dest] for dest in destinations)) <= capacity[fac] * y[fac])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.2f}\n')
    print('Facilities opened:')
    for fac in facilities:
        if y[fac].X > 0.5:
            print(f'  {fac} (Fixed cost: {fixed_cost[fac]}, Capacity: {capacity[fac]})')
    print('\nShipments:')
    for fac in facilities:
        for dest in destinations:
            if x[fac, dest].X > 1e-06:
                print(f'  {fac} -> {dest}: {x[fac, dest].X:.2f} units (Cost per unit: {shipping_cost[fac][dest]})')
else:
    print(f'No optimal solution found. Status: {m.status}')