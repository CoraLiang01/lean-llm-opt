import gurobipy as gp
import pandas as pd
import numpy as np
import re
facility_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/facility_costs.csv'
shipping_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/shipping_costs.csv'
demand_requirements_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/demand_requirements.csv'
fac_df = pd.read_csv(facility_costs_path, sep=',')
fac_df['Facility'] = fac_df['Facility'].astype(str).str.strip()
ship_df = pd.read_csv(shipping_costs_path, sep=',')
ship_df['Origin'] = ship_df['Origin'].astype(str).str.strip()
dem_df = pd.read_csv(demand_requirements_path, sep=',')
dem_df['Destination'] = dem_df['Destination'].astype(str).str.strip()
facilities = [f'A{i}' for i in range(1, 16)]
distribution_centers = [f'B{j}' for j in range(1, 9)]
missing_fac = set(facilities) - set(fac_df['Facility'])
if missing_fac:
    raise ValueError(f'Missing facilities in facility_costs.csv: {missing_fac}')
missing_ship_fac = set(facilities) - set(ship_df['Origin'])
if missing_ship_fac:
    raise ValueError(f'Missing facilities in shipping_costs.csv: {missing_ship_fac}')
missing_dcs = set(distribution_centers) - set(dem_df['Destination'])
if missing_dcs:
    raise ValueError(f'Missing distribution centers in demand_requirements.csv: {missing_dcs}')
missing_ship_dcs = set(distribution_centers) - set(ship_df.columns[1:])
if missing_ship_dcs:
    raise ValueError(f'Missing distribution centers in shipping_costs.csv columns: {missing_ship_dcs}')
fixed_cost = fac_df.set_index('Facility')['FixedCost'].astype(float).to_dict()
capacity = fac_df.set_index('Facility')['Capacity'].astype(float).to_dict()
shipping_cost = {}
for (_, row) in ship_df.iterrows():
    i = str(row['Origin']).strip()
    for j in distribution_centers:
        shipping_cost[i, j] = float(row[j])
demand = dem_df.set_index('Destination')['Demand'].astype(float).to_dict()
m = gp.Model('UFLP')
y = m.addVars(facilities, vtype=gp.GRB.BINARY, name='')
x = m.addVars(facilities, distribution_centers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
m.setObjective(gp.quicksum((fixed_cost[i] * y[i] for i in facilities)) + gp.quicksum((shipping_cost[i, j] * x[i, j] for i in facilities for j in distribution_centers)), gp.GRB.MINIMIZE)
for j in distribution_centers:
    m.addConstr(gp.quicksum((x[i, j] for i in facilities)) == demand[j], name=f'demand_{j}')
for i in facilities:
    m.addConstr(gp.quicksum((x[i, j] for j in distribution_centers)) <= capacity[i] * y[i], name=f'capacity_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('\n--- Facility Construction Decisions ---')
    for i in facilities:
        if y[i].X > 0.5:
            print(f'  Facility {i}: BUILT (FixedCost={fixed_cost[i]:.2f}, Capacity={capacity[i]:.0f})')
        else:
            print(f'  Facility {i}: NOT BUILT')
    print('\n--- Shipment Plan ---')
    for j in distribution_centers:
        print(f'  Distribution Center {j} (Demand={demand[j]:.0f}):')
        for i in facilities:
            shipped = x[i, j].X
            if shipped > 1e-06:
                print(f'    From {i}: {shipped:.2f} units (ShippingCost={shipping_cost[i, j]:.2f}/unit)')
else:
    print(f'No optimal solution found. Status: {m.status}')