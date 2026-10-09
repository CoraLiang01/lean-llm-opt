import gurobipy as gp
import pandas as pd
import numpy as np
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/products.csv'
capacity_df = pd.read_csv(capacity_path, sep=',', dtype=str, keep_default_na=False)
products_df = pd.read_csv(products_path, sep=',', dtype=str, keep_default_na=False)
capacity_df['VehicleType_norm'] = capacity_df['VehicleType'].str.strip().str.casefold()
products_df['ProductName_norm'] = products_df['ProductName'].str.strip().str.casefold()
merged_df = pd.merge(capacity_df, products_df, left_on='VehicleType_norm', right_on='ProductName_norm', how='inner', suffixes=('_cap', '_prod'))
if merged_df.shape[0] != capacity_df.shape[0]:
    missing = set(capacity_df['VehicleType_norm']) - set(merged_df['VehicleType_norm'])
    raise ValueError(f'Missing benefit coefficients for vehicle types: {missing}')
vehicle_types = list(merged_df['VehicleType'])
capacity_dict = {}
value_dict = {}
for (_, row) in merged_df.iterrows():
    vt = row['VehicleType']
    try:
        cap = int(row['Capacity'])
        val = int(row['Value'])
    except Exception as e:
        raise ValueError(f"Non-integer value in capacity or value for vehicle type '{vt}': {e}")
    capacity_dict[vt] = cap
    value_dict[vt] = val
total_inventory_capacity = sum((capacity_dict[vt] for vt in vehicle_types))
m = gp.Model('NewCarSalesInventory')
x_vars = m.addVars(vehicle_types, vtype=gp.GRB.INTEGER, lb=0, ub=[capacity_dict[vt] for vt in vehicle_types], name='')
m.setObjective(gp.quicksum((value_dict[vt] * x_vars[vt] for vt in vehicle_types)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((x_vars[vt] for vt in vehicle_types)) <= total_inventory_capacity, name='total_inventory')
m.addConstrs((x_vars[vt] <= capacity_dict[vt] for vt in vehicle_types), name='')
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for vt in vehicle_types:
        print(f'{x_vars[vt].VarName} {x_vars[vt].X}')
else:
    print(f'Solver status: {m.status}')