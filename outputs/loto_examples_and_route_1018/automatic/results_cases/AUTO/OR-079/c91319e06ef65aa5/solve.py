import gurobipy as gp
import pandas as pd
import numpy as np
import re
facility_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/facility_costs.csv'
shipping_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/shipping_costs.csv'
demand_requirements_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/demand_requirements.csv'
facility_df = pd.read_csv(facility_costs_path, dtype=str, keep_default_na=False)
shipping_df = pd.read_csv(shipping_costs_path, dtype=str, keep_default_na=False)
demand_df = pd.read_csv(demand_requirements_path, dtype=str, keep_default_na=False)
facilities = facility_df['Facility'].tolist()
distribution_centers = demand_df['Destination'].tolist()
facility_fixed_cost = {}
facility_capacity = {}
for idx, row in facility_df.iterrows():
    fac = row['Facility']
    facility_fixed_cost[fac] = int(row['FixedCost'])
    facility_capacity[fac] = int(row['Capacity'])
demand = {}
for idx, row in demand_df.iterrows():
    dc = row['Destination']
    demand[dc] = int(row['Demand'])
shipping_cost = {}
shipping_facilities = shipping_df['Origin'].tolist()
shipping_dc_columns = [col for col in shipping_df.columns if col != 'Origin']
missing_facilities = set(facilities) - set(shipping_facilities)
if missing_facilities:
    raise ValueError(f'Facilities missing in shipping_costs.csv: {missing_facilities}')
missing_dcs = set(distribution_centers) - set(shipping_dc_columns)
if missing_dcs:
    raise ValueError(f'Distribution centers missing in shipping_costs.csv: {missing_dcs}')
for idx, row in shipping_df.iterrows():
    fac = row['Origin']
    for dc in distribution_centers:
        shipping_cost[fac, dc] = int(row[dc])
m = gp.Model('CapacitatedFacilityLocation')
y_vars = m.addVars(facilities, vtype=gp.GRB.BINARY, name='')
x_vars = m.addVars(facilities, distribution_centers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
m.setObjective(gp.quicksum((facility_fixed_cost[i] * y_vars[i] for i in facilities)) + gp.quicksum((shipping_cost[i, j] * x_vars[i, j] for i in facilities for j in distribution_centers)), gp.GRB.MINIMIZE)
for j in distribution_centers:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in facilities)) == demand[j], name=f'demand_{j}')
for i in facilities:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in distribution_centers)) <= facility_capacity[i] * y_vars[i], name=f'capacity_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('\n--- Facility Construction Decisions ---')
    for i in facilities:
        if y_vars[i].X > 0.5:
            print(f'  Facility {i}: Constructed (FixedCost={facility_fixed_cost[i]})')
        else:
            print(f'  Facility {i}: Not Constructed')
    print('\n--- Shipment Plan ---')
    for i in facilities:
        for j in distribution_centers:
            shipped = x_vars[i, j].X
            if shipped > 1e-06:
                print(f'  Ship {shipped:.2f} units from {i} to {j} (Cost/unit={shipping_cost[i, j]})')
else:
    print(f'No optimal solution found. Status: {m.status}')