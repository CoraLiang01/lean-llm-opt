import gurobipy as gp
import pandas as pd
import numpy as np
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/products.csv'
df_capacity = pd.read_csv(capacity_path, sep=',')
df_products = pd.read_csv(products_path, sep=',')
df_capacity['VehicleType_norm'] = df_capacity['VehicleType'].astype(str).str.strip().str.casefold()
df_products['ProductName_norm'] = df_products['ProductName'].astype(str).str.strip().str.casefold()
merged = pd.merge(df_capacity, df_products, left_on='VehicleType_norm', right_on='ProductName_norm', how='inner', suffixes=('_cap', '_prod'))
if merged.shape[0] == 0:
    raise ValueError('No matching vehicle types found between capacity.csv and products.csv.')
vehicle_types = merged['VehicleType'].tolist()
benefit = dict(zip(merged['VehicleType'], merged['Value']))
capacity = dict(zip(merged['VehicleType'], merged['Capacity']))
total_inventory_capacity = sum(capacity.values())
m = gp.Model('NewCarSalesInventory')
x = m.addVars(vehicle_types, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((benefit[i] * x[i] for i in vehicle_types)), gp.GRB.MAXIMIZE)
m.addConstrs((x[i] <= capacity[i] for i in vehicle_types), name='')
m.addConstr(gp.quicksum((x[i] for i in vehicle_types)) <= total_inventory_capacity, name='total_cap')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total benefit: {m.objVal:.0f}')
    print('--- Optimal Daily Order Plan ---')
    for i in vehicle_types:
        print(f'  {i}: {int(round(x[i].X))} units (max {capacity[i]})')
else:
    print(f'No optimal solution found. Status: {m.status}')