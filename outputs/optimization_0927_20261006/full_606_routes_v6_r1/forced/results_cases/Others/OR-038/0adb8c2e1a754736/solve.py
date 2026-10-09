import gurobipy as gp
import pandas as pd
import numpy as np
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/products.csv'
capacity_df = pd.read_csv(capacity_path, dtype=str, keep_default_na=False)
products_df = pd.read_csv(products_path, dtype=str, keep_default_na=False)
capacity_df['VehicleType_norm'] = capacity_df['VehicleType'].str.strip().str.casefold()
products_df['ProductName_norm'] = products_df['ProductName'].str.strip().str.casefold()
vehicle_types = list(capacity_df['VehicleType'])
norm_to_vehicle = dict(zip(capacity_df['VehicleType_norm'], capacity_df['VehicleType']))
capacity_dict = {}
for (idx, row) in capacity_df.iterrows():
    vt = row['VehicleType']
    try:
        cap = int(row['Capacity'])
    except Exception:
        raise ValueError(f"Invalid Capacity value for VehicleType '{vt}': {row['Capacity']}")
    capacity_dict[vt] = cap
value_dict = {}
for (idx, row) in products_df.iterrows():
    pname_norm = row['ProductName_norm']
    if pname_norm in norm_to_vehicle:
        vt = norm_to_vehicle[pname_norm]
        try:
            val = int(row['Value'])
        except Exception:
            raise ValueError(f"Invalid Value for ProductName '{row['ProductName']}': {row['Value']}")
        value_dict[vt] = val
missing_value_types = [vt for vt in vehicle_types if vt not in value_dict]
if missing_value_types:
    raise ValueError(f'Missing Value in products.csv for VehicleType(s): {missing_value_types}')
m = gp.Model('NewCarSalesInventoryReplenishment')
x_vars = m.addVars(vehicle_types, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[vt] * x_vars[vt] for vt in vehicle_types)), gp.GRB.MAXIMIZE)
for vt in vehicle_types:
    m.addConstr(x_vars[vt] <= capacity_dict[vt], name=f'cap_{vt}')
total_capacity = sum((capacity_dict[vt] for vt in vehicle_types))
m.addConstr(gp.quicksum((x_vars[vt] for vt in vehicle_types)) <= total_capacity, name='total_capacity')
m.optimize()