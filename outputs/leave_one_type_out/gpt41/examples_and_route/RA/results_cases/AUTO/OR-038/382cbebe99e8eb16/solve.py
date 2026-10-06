import gurobipy as gp
import pandas as pd
import numpy as np
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/products.csv'
df_capacity = pd.read_csv(capacity_path, sep=',')
df_products = pd.read_csv(products_path, sep=',')
df_capacity['VehicleType_norm'] = df_capacity['VehicleType'].str.casefold().str.strip()
df_products['ProductName_norm'] = df_products['ProductName'].str.casefold().str.strip()
df_merged = pd.merge(df_capacity, df_products, left_on='VehicleType_norm', right_on='ProductName_norm', how='inner', suffixes=('_cap', '_prod'))
if len(df_merged) != len(df_capacity):
    missing = set(df_capacity['VehicleType_norm']) - set(df_merged['VehicleType_norm'])
    raise ValueError(f'Missing benefit coefficients for vehicle types: {missing}')
vehicle_types = list(df_merged['VehicleType'])
benefit = dict(zip(df_merged['VehicleType'], df_merged['Value']))
capacity = dict(zip(df_merged['VehicleType'], df_merged['Capacity']))
total_inventory_capacity = sum(capacity.values())
m = gp.Model('NewCarSalesInventory')
x = m.addVars(vehicle_types, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((benefit[i] * x[i] for i in vehicle_types)), gp.GRB.MAXIMIZE)
m.addConstrs((x[i] <= capacity[i] for i in vehicle_types), name='')
m.addConstr(gp.quicksum((x[i] for i in vehicle_types)) <= total_inventory_capacity, name='total_inventory_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total benefit: {m.objVal:.0f}')
    print('--- Optimal Daily Order Plan ---')
    for i in vehicle_types:
        print(f'  {i}: {int(round(x[i].X))} units (max {capacity[i]})')
else:
    print(f'No optimal solution found. Status: {m.status}')