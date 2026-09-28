import gurobipy as gp
import pandas as pd
import numpy as np
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/products.csv'
df_capacity = pd.read_csv(capacity_path, sep=',')
df_products = pd.read_csv(products_path, sep=',')
df_capacity['VehicleType_norm'] = df_capacity['VehicleType'].str.strip().str.casefold()
df_products['ProductName_norm'] = df_products['ProductName'].str.strip().str.casefold()
df_merged = pd.merge(df_capacity, df_products, left_on='VehicleType_norm', right_on='ProductName_norm', how='inner', suffixes=('_cap', '_prod'))
if df_merged.shape[0] != df_capacity.shape[0]:
    missing = set(df_capacity['VehicleType_norm']) - set(df_merged['VehicleType_norm'])
    raise ValueError(f'Missing benefit coefficients for vehicle types: {missing}')
vehicle_ids = df_merged['VehicleID'].tolist()
vehicle_types = df_merged['VehicleType'].tolist()
vehicle_id_to_type = dict(zip(vehicle_ids, vehicle_types))
value = dict(zip(df_merged['VehicleID'], df_merged['Value']))
capacity = dict(zip(df_merged['VehicleID'], df_merged['Capacity']))
total_inventory_capacity = sum(capacity.values())
m = gp.Model('NewCarSalesInventory')
x = m.addVars(vehicle_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value[i] * x[i] for i in vehicle_ids)), gp.GRB.MAXIMIZE)
m.addConstrs((x[i] <= capacity[i] for i in vehicle_ids), name='')
m.addConstr(gp.quicksum((x[i] for i in vehicle_ids)) <= total_inventory_capacity, name='total_cap')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total benefit: {m.objVal:.0f}')
    print('Order plan per vehicle type:')
    for i in vehicle_ids:
        print(f'  {vehicle_id_to_type[i]}: {int(round(x[i].X))} units (limit: {capacity[i]})')
else:
    print(f'No optimal solution found. Status: {m.status}')