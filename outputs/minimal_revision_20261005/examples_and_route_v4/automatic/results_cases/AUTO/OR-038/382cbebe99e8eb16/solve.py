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
vehicle_keys = list(df_merged['VehicleID'])
vehicle_types = dict(zip(df_merged['VehicleID'], df_merged['VehicleType']))
benefit = dict(zip(df_merged['VehicleID'], df_merged['Value']))
capacity = dict(zip(df_merged['VehicleID'], df_merged['Capacity']))
total_inventory_capacity = sum(capacity.values())

def solve_problem(vehicle_keys, benefit, capacity, total_inventory_capacity):
    m = gp.Model('NewCarSalesInventory')
    x = m.addVars(vehicle_keys, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((benefit[i] * x[i] for i in vehicle_keys)), gp.GRB.MAXIMIZE)
    m.addConstrs((x[i] <= capacity[i] for i in vehicle_keys), name='')
    m.addConstr(gp.quicksum((x[i] for i in vehicle_keys)) <= total_inventory_capacity, name='totalcap')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    return m
m = solve_problem(vehicle_keys, benefit, capacity, total_inventory_capacity)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')