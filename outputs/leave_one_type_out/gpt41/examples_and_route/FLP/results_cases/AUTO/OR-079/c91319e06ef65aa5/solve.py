import gurobipy as gp
import pandas as pd
import numpy as np
import re
facility_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/facility_costs.csv'
shipping_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/shipping_costs.csv'
demand_requirements_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/demand_requirements.csv'
fac_df = pd.read_csv(facility_costs_path, sep=',')
fac_df['Facility'] = fac_df['Facility'].astype(str).str.strip().str.upper()
facilities = fac_df['Facility'].tolist()
dem_df = pd.read_csv(demand_requirements_path, sep=',')
dem_df['Destination'] = dem_df['Destination'].astype(str).str.strip().str.upper()
dcs = dem_df['Destination'].tolist()
ship_df = pd.read_csv(shipping_costs_path, sep=',')
ship_df['Origin'] = ship_df['Origin'].astype(str).str.strip().str.upper()
fixed_cost = {}
capacity = {}
for _, row in fac_df.iterrows():
    fac = row['Facility']
    fixed_cost[fac] = float(row['FixedCost'])
    capacity[fac] = float(row['Capacity'])
demand = {}
for _, row in dem_df.iterrows():
    dc = row['Destination']
    demand[dc] = float(row['Demand'])
shipping_cost = {}
for _, row in ship_df.iterrows():
    fac = row['Origin']
    for dc in dcs:
        if dc not in row:
            raise KeyError(f'Shipping cost missing for facility {fac} to DC {dc}')
        shipping_cost[fac, dc] = float(row[dc])
if set(facilities) != set(ship_df['Origin']):
    raise ValueError('Mismatch between facilities in facility_costs.csv and shipping_costs.csv')
if set(dcs) != set(ship_df.columns[1:]):
    raise ValueError('Mismatch between DCs in demand_requirements.csv and shipping_costs.csv columns')
m = gp.Model('CapacitatedFacilityLocation')
y = m.addVars(facilities, vtype=gp.GRB.BINARY, name='')
x = m.addVars(facilities, dcs, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
m.setObjective(gp.quicksum((fixed_cost[i] * y[i] for i in facilities)) + gp.quicksum((shipping_cost[i, j] * x[i, j] for i in facilities for j in dcs)), gp.GRB.MINIMIZE)
for j in dcs:
    m.addConstr(gp.quicksum((x[i, j] for i in facilities)) == demand[j], name=f'demand_{j}')
for i in facilities:
    m.addConstr(gp.quicksum((x[i, j] for j in dcs)) <= capacity[i] * y[i], name=f'cap_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('\n--- Facility Construction Decisions ---')
    for i in facilities:
        if y[i].X > 0.5:
            print(f'  Facility {i}: BUILT (FixedCost={fixed_cost[i]:.2f}, Capacity={capacity[i]:.0f})')
        else:
            print(f'  Facility {i}: NOT BUILT')
    print('\n--- Shipment Plan (facility -> DC: amount) ---')
    for i in facilities:
        for j in dcs:
            if x[i, j].X > 1e-06:
                print(f'  {i} -> {j}: {x[i, j].X:.2f} units (Cost/unit={shipping_cost[i, j]:.2f})')
else:
    print(f'No optimal solution found. Status: {m.status}')