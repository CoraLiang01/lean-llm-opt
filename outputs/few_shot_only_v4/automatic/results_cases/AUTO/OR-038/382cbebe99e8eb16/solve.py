import pandas as pd
import gurobipy as gp
from gurobipy import GRB
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/products.csv'
capacity_df = pd.read_csv(capacity_path, dtype=str, keep_default_na=False)
products_df = pd.read_csv(products_path, dtype=str, keep_default_na=False)
vehicle_types_capacity = list(capacity_df['VehicleType'])
vehicle_types_products = list(products_df['VehicleType'])
if set(vehicle_types_capacity) != set(vehicle_types_products):
    raise ValueError('Mismatch between vehicle types in capacity.csv and products.csv')
vehicle_types = vehicle_types_capacity
try:
    capacity_dict = {row['VehicleType']: int(row['Capacity']) for _, row in capacity_df.iterrows()}
except Exception as e:
    raise ValueError(f'Error parsing Capacity from capacity.csv: {e}')
try:
    benefit_dict = {row['VehicleType']: float(row['Value']) for _, row in products_df.iterrows()}
except Exception as e:
    raise ValueError(f'Error parsing Value from products.csv: {e}')
total_capacity = sum((capacity_dict[vt] for vt in vehicle_types))
m = gp.Model('NewCarSalesInNorway1')
x_vars = m.addVars(vehicle_types, lb=0, ub=[capacity_dict[vt] for vt in vehicle_types], vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((benefit_dict[vt] * x_vars[vt] for vt in vehicle_types)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((x_vars[vt] for vt in vehicle_types)) <= total_capacity, name='total_inventory')
for vt in vehicle_types:
    m.addConstr(x_vars[vt] >= 0, name=f'nonneg_{vt}')
    m.addConstr(x_vars[vt] <= capacity_dict[vt], name=f'cap_{vt}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for vt in vehicle_types:
        print(f'{x_vars[vt].VarName}: {x_vars[vt].X}')
else:
    print(f'Solver status: {m.Status}')