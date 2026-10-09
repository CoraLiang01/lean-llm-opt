import gurobipy as gp
import pandas as pd
import numpy as np
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/products.csv'
capacity_df = pd.read_csv(capacity_path, sep=',', dtype=str, keep_default_na=False)
products_df = pd.read_csv(products_path, sep=',', dtype=str, keep_default_na=False)

def norm_str(s):
    return s.strip().casefold()
vehicle_types = capacity_df['VehicleType'].apply(norm_str).tolist()
vehicle_type_to_id = {norm_str(row['VehicleType']): row['VehicleID'] for (_, row) in capacity_df.iterrows()}
capacity_dict = {}
for (_, row) in capacity_df.iterrows():
    vt = norm_str(row['VehicleType'])
    try:
        cap = int(row['Capacity'])
    except Exception:
        raise ValueError(f"Invalid Capacity value for VehicleType '{row['VehicleType']}': {row['Capacity']}")
    capacity_dict[vt] = cap
value_dict = {}
for (_, row) in products_df.iterrows():
    pn = norm_str(row['ProductName'])
    try:
        val = int(row['Value'])
    except Exception:
        raise ValueError(f"Invalid Value for ProductName '{row['ProductName']}': {row['Value']}")
    value_dict[pn] = val
for vt in vehicle_types:
    if vt not in value_dict:
        raise KeyError(f"VehicleType '{vt}' in capacity.csv not found in products.csv ProductName column.")
I = vehicle_types
capacity = {vt: capacity_dict[vt] for vt in I}
value = {vt: value_dict[vt] for vt in I}
total_inventory_capacity = sum((capacity[vt] for vt in I))
m = gp.Model('NewCarSalesInventory')
x_vars = m.addVars(I, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value[vt] * x_vars[vt] for vt in I)), gp.GRB.MAXIMIZE)
m.addConstrs((x_vars[vt] <= capacity[vt] for vt in I), name='')
m.addConstr(gp.quicksum((x_vars[vt] for vt in I)) <= total_inventory_capacity, name='total_inventory_capacity')
m.optimize()