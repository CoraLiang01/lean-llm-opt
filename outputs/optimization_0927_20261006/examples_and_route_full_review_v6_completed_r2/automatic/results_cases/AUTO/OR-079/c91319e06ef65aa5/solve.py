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
for (idx, row) in facility_df.iterrows():
    fac = row['Facility']
    try:
        facility_fixed_cost[fac] = int(row['FixedCost'])
        facility_capacity[fac] = int(row['Capacity'])
    except Exception as e:
        raise ValueError(f"Invalid numeric value in facility_costs.csv for Facility '{fac}': {e}")
demand = {}
for (idx, row) in demand_df.iterrows():
    dc = row['Destination']
    try:
        demand[dc] = int(row['Demand'])
    except Exception as e:
        raise ValueError(f"Invalid numeric value in demand_requirements.csv for Destination '{dc}': {e}")
shipping_cost = {}
shipping_facilities = shipping_df['Origin'].tolist()
shipping_columns = [col for col in shipping_df.columns if col != 'Origin']
if set(facilities) != set(shipping_facilities):
    missing_in_shipping = set(facilities) - set(shipping_facilities)
    missing_in_facilities = set(shipping_facilities) - set(facilities)
    raise ValueError(f'Mismatch between facilities in facility_costs.csv and shipping_costs.csv. Missing in shipping_costs.csv: {missing_in_shipping}. Extra in shipping_costs.csv: {missing_in_facilities}.')
if set(distribution_centers) != set(shipping_columns):
    missing_in_shipping = set(distribution_centers) - set(shipping_columns)
    missing_in_demand = set(shipping_columns) - set(distribution_centers)
    raise ValueError(f'Mismatch between distribution centers in demand_requirements.csv and shipping_costs.csv. Missing in shipping_costs.csv: {missing_in_shipping}. Extra in shipping_costs.csv: {missing_in_demand}.')
for (idx, row) in shipping_df.iterrows():
    fac = row['Origin']
    for dc in distribution_centers:
        try:
            shipping_cost[fac, dc] = int(row[dc])
        except Exception as e:
            raise ValueError(f"Invalid numeric value in shipping_costs.csv for Origin '{fac}', Destination '{dc}': {e}")
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
        print(f"Facility {i}: {('OPEN' if y_vars[i].X > 0.5 else 'CLOSED')} (y={int(round(y_vars[i].X))})")
    print('\n--- Shipment Plan ---')
    for i in facilities:
        for j in distribution_centers:
            shipped = x_vars[i, j].X
            if shipped > 1e-06:
                print(f'  Ship {shipped:.2f} units from {i} to {j} (cost per unit: {shipping_cost[i, j]})')
else:
    print(f'No optimal solution found. Status: {m.status}')