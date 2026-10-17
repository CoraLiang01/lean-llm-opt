import gurobipy as gp
import pandas as pd
import numpy as np
import re
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/products.csv'
capacity_df = pd.read_csv(capacity_path, sep=',', dtype=str, keep_default_na=False)
products_df = pd.read_csv(products_path, sep=',', dtype=str, keep_default_na=False)

def norm_str(s):
    return s.strip().casefold()
capacity_df['VehicleType_norm'] = capacity_df['VehicleType'].apply(norm_str)
products_df['ProductName_norm'] = products_df['ProductName'].apply(norm_str)
merged_df = pd.merge(capacity_df, products_df, left_on='VehicleType_norm', right_on='ProductName_norm', how='inner', suffixes=('_cap', '_prod'))
if merged_df.shape[0] != capacity_df.shape[0]:
    missing = set(capacity_df['VehicleType_norm']) - set(merged_df['VehicleType_norm'])
    raise ValueError(f'Missing benefit coefficients for vehicle types: {missing}')
vehicle_types = list(merged_df['VehicleType'])
benefit_coeff = {}
capacity_limit = {}
for idx, row in merged_df.iterrows():
    vt = row['VehicleType']
    try:
        benefit_coeff[vt] = int(row['Value'])
    except Exception:
        raise ValueError(f"Invalid Value for vehicle type '{vt}': {row['Value']}")
    try:
        capacity_limit[vt] = int(row['Capacity'])
    except Exception:
        raise ValueError(f"Invalid Capacity for vehicle type '{vt}': {row['Capacity']}")
total_inventory_capacity = sum((capacity_limit[vt] for vt in vehicle_types))
m = gp.Model('NewCarSalesInventory')
x_vars = m.addVars(vehicle_types, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((benefit_coeff[vt] * x_vars[vt] for vt in vehicle_types)), gp.GRB.MAXIMIZE)
for vt in vehicle_types:
    m.addConstr(x_vars[vt] <= capacity_limit[vt], name=f'cap_{vt}')
m.addConstr(gp.quicksum((x_vars[vt] for vt in vehicle_types)) <= total_inventory_capacity, name='total_inventory')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total benefit: {m.objVal:.0f}')
    print('\n--- Optimal Daily Order Plan ---')
    for vt in vehicle_types:
        print(f'  {vt}: {int(round(x_vars[vt].X))} units (limit: {capacity_limit[vt]})')
else:
    print(f'No optimal solution found. Status: {m.status}')