import gurobipy as gp
import pandas as pd
import numpy as np
import re
facility_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/facility_costs.csv'
shipping_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/shipping_costs.csv'
demand_requirements_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/demand_requirements.csv'
df_fac = pd.read_csv(facility_costs_path, sep=',')
facilities = df_fac['Facility'].astype(str).str.strip().tolist()
df_dem = pd.read_csv(demand_requirements_path, sep=',')
dist_centers = df_dem['Destination'].astype(str).str.strip().tolist()
fixed_cost = {}
capacity = {}
for _, row in df_fac.iterrows():
    fac = str(row['Facility']).strip()
    fixed_cost[fac] = int(row['FixedCost'])
    capacity[fac] = int(row['Capacity'])
demand = {}
for _, row in df_dem.iterrows():
    dc = str(row['Destination']).strip()
    demand[dc] = int(row['Demand'])
df_ship = pd.read_csv(shipping_costs_path, sep=',')
shipping_cost = {}
for _, row in df_ship.iterrows():
    fac = str(row['Origin']).strip()
    if fac not in facilities:
        continue
    for dc in dist_centers:
        if dc not in row:
            raise KeyError(f'Shipping cost missing for facility {fac} to DC {dc}')
        shipping_cost[fac, dc] = float(row[dc])
if set(facilities) != set(df_ship['Origin'].astype(str).str.strip()):
    raise ValueError('Mismatch between facilities in facility_costs.csv and shipping_costs.csv')
if set(dist_centers) != set(df_ship.columns[1:]):
    raise ValueError('Mismatch between distribution centers in shipping_costs.csv and demand_requirements.csv')
m = gp.Model('CapacitatedFacilityLocation')
y = m.addVars(facilities, vtype=gp.GRB.BINARY, name='')
x = m.addVars(facilities, dist_centers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
m.setObjective(gp.quicksum((fixed_cost[i] * y[i] for i in facilities)) + gp.quicksum((shipping_cost[i, j] * x[i, j] for i in facilities for j in dist_centers)), gp.GRB.MINIMIZE)
for j in dist_centers:
    m.addConstr(gp.quicksum((x[i, j] for i in facilities)) == demand[j], name=f'demand_{j}')
for i in facilities:
    m.addConstr(gp.quicksum((x[i, j] for j in dist_centers)) <= capacity[i] * y[i], name=f'capacity_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('\n--- Facility Construction Decisions ---')
    for i in facilities:
        if y[i].X > 0.5:
            print(f'  Facility {i}: BUILT (Fixed cost: {fixed_cost[i]}, Capacity: {capacity[i]})')
        else:
            print(f'  Facility {i}: NOT BUILT')
    print('\n--- Shipment Plan ---')
    for i in facilities:
        if y[i].X > 0.5:
            for j in dist_centers:
                shipped = x[i, j].X
                if shipped > 1e-06:
                    print(f'  Ship {shipped:.2f} units from {i} to {j} (Cost per unit: {shipping_cost[i, j]})')
else:
    print(f'No optimal solution found. Status: {m.status}')