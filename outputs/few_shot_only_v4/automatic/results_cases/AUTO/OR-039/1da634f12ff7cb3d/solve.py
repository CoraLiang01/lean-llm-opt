import pandas as pd
import gurobipy as gp
from gurobipy import GRB
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/products.csv', dtype=str, keep_default_na=False)
if not {'ProductName', 'Value', 'Weight'}.issubset(products_df.columns):
    raise ValueError('products.csv must contain columns: ProductName, Value, Weight')
vehicle_types = products_df['ProductName'].tolist()
p_i = {}
w_i = {}
for _, row in products_df.iterrows():
    try:
        p_i[row['ProductName']] = float(row['Value'])
        w_i[row['ProductName']] = float(row['Weight'])
    except Exception as e:
        raise ValueError(f"Invalid Value or Weight for product {row['ProductName']}: {e}")
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/capacity.csv', dtype=str, keep_default_na=False)
if not {'Warehouse ID', 'Capacity'}.issubset(capacity_df.columns):
    raise ValueError('capacity.csv must contain columns: Warehouse ID, Capacity')
warehouses = capacity_df['Warehouse ID'].tolist()
C_j = {}
for _, row in capacity_df.iterrows():
    try:
        C_j[row['Warehouse ID']] = float(row['Capacity'])
    except Exception as e:
        raise ValueError(f"Invalid Capacity for warehouse {row['Warehouse ID']}: {e}")
if set(vehicle_types) != set(p_i.keys()) or set(vehicle_types) != set(w_i.keys()):
    raise ValueError('Mismatch in vehicle types and their coefficients in products.csv')
if set(warehouses) != set(C_j.keys()):
    raise ValueError('Mismatch in warehouses and their capacities in capacity.csv')
m = gp.Model('NewCarSalesInNorway2')
x_vars = m.addVars(vehicle_types, warehouses, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((p_i[i] * x_vars[i, j] for i in vehicle_types for j in warehouses)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((w_i[i] * x_vars[i, j] for i in vehicle_types)) <= C_j[j] for j in warehouses), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')