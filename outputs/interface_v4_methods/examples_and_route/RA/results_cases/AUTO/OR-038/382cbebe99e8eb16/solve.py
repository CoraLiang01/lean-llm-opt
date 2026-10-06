import gurobipy as gp
import pandas as pd
import numpy as np
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/products.csv'
capacity_df = pd.read_csv(capacity_path, sep=',')
products_df = pd.read_csv(products_path, sep=',')

def norm(s):
    return str(s).strip().casefold()
capacity_df['VehicleType_norm'] = capacity_df['VehicleType'].apply(norm)
products_df['ProductName_norm'] = products_df['ProductName'].apply(norm)
merged = pd.merge(capacity_df, products_df, left_on='VehicleType_norm', right_on='ProductName_norm', how='inner', validate='one_to_one')
if len(merged) != len(capacity_df):
    missing = set(capacity_df['VehicleType_norm']) - set(merged['VehicleType_norm'])
    raise ValueError(f'Missing benefit coefficients for vehicle types: {missing}')
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