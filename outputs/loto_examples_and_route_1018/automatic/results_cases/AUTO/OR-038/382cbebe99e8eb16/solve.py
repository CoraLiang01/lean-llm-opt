import gurobipy as gp
import pandas as pd
import numpy as np
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/products.csv'
capacity_df = pd.read_csv(capacity_path, sep=',', dtype=str, keep_default_na=False)
products_df = pd.read_csv(products_path, sep=',', dtype=str, keep_default_na=False)
capacity_df['VehicleType_norm'] = capacity_df['VehicleType'].str.strip().str.casefold()
products_df['ProductName_norm'] = products_df['ProductName'].str.strip().str.casefold()
vehicle_types = list(capacity_df['VehicleType'])
norm_to_vehicle = dict(zip(capacity_df['VehicleType_norm'], capacity_df['VehicleType']))
capacity_dict = {}
for idx, row in capacity_df.iterrows():
    vt = row['VehicleType']
    try:
        cap = int(row['Capacity'])
    except Exception:
        raise ValueError(f"Invalid Capacity value for VehicleType '{vt}': {row['Capacity']}")
    capacity_dict[vt] = cap
value_dict = {}
for idx, row in capacity_df.iterrows():
    vt = row['VehicleType']
    vt_norm = row['VehicleType_norm']
    match = products_df[products_df['ProductName_norm'] == vt_norm]
    if len(match) != 1:
        raise ValueError(f"Could not find unique matching product for VehicleType '{vt}' (normalized '{vt_norm}'). Found {len(match)} matches.")
    value_str = match.iloc[0]['Value']
    try:
        value = int(value_str)
    except Exception:
        raise ValueError(f"Invalid Value for VehicleType '{vt}': {value_str}")
    value_dict[vt] = value
if set(vehicle_types) != set(capacity_dict.keys()) or set(vehicle_types) != set(value_dict.keys()):
    raise ValueError('Mismatch in vehicle_types, capacity_dict, or value_dict keys.')
total_inventory_capacity = sum((capacity_dict[vt] for vt in vehicle_types))
m = gp.Model('NewCarSalesInventory')
x_vars = m.addVars(vehicle_types, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[vt] * x_vars[vt] for vt in vehicle_types)), gp.GRB.MAXIMIZE)
for vt in vehicle_types:
    m.addConstr(x_vars[vt] <= capacity_dict[vt], name=f'cap_{vt}')
m.addConstr(gp.quicksum((x_vars[vt] for vt in vehicle_types)) <= total_inventory_capacity, name='total_inventory')
m.optimize()