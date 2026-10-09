import gurobipy as gp
import pandas as pd
import numpy as np
import re
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/products.csv'
capacity_df = pd.read_csv(capacity_path, sep=',', dtype=str, keep_default_na=False)
products_df = pd.read_csv(products_path, sep=',', dtype=str, keep_default_na=False)

def norm(s):
    return re.sub('\\s+', ' ', s.strip()).casefold()
capacity_df['VehicleType_norm'] = capacity_df['VehicleType'].apply(norm)
products_df['ProductName_norm'] = products_df['ProductName'].apply(norm)
merged_df = pd.merge(capacity_df, products_df, left_on='VehicleType_norm', right_on='ProductName_norm', how='inner', suffixes=('_cap', '_prod'))
if merged_df.shape[0] != capacity_df.shape[0]:
    missing = set(capacity_df['VehicleType_norm']) - set(merged_df['VehicleType_norm'])
    raise ValueError(f'Missing product data for vehicle types: {missing}')
vehicle_types = list(merged_df['VehicleType'])
capacity_dict = dict(zip(merged_df['VehicleType'], merged_df['Capacity'].astype(int)))
value_dict = dict(zip(merged_df['VehicleType'], merged_df['Value'].astype(int)))
total_inventory_capacity = sum(capacity_dict.values())
m = gp.Model('NewCarSalesInventory')
x_vars = m.addVars(vehicle_types, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[vt] * x_vars[vt] for vt in vehicle_types)), gp.GRB.MAXIMIZE)
for vt in vehicle_types:
    m.addConstr(x_vars[vt] <= capacity_dict[vt], name=f'cap_{vt}')
m.addConstr(gp.quicksum((x_vars[vt] for vt in vehicle_types)) <= total_inventory_capacity, name='total_cap')
m.optimize()