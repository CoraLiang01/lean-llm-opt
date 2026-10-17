import pandas as pd
import gurobipy as gp
from gurobipy import GRB
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/products.csv', dtype=str, keep_default_na=False)
products_df['Value'] = products_df['Value'].astype(float)
products_df['Weight'] = products_df['Weight'].astype(float)
vehicle_types = products_df['ProductName'].tolist()
p = dict(zip(products_df['ProductName'], products_df['Value']))
w = dict(zip(products_df['ProductName'], products_df['Weight']))
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/capacity.csv', dtype=str, keep_default_na=False, delim_whitespace=False)
capacity_df['Capacity'] = capacity_df['Capacity'].astype(float)
warehouses = capacity_df['Warehouse ID'].tolist()
C = dict(zip(capacity_df['Warehouse ID'], capacity_df['Capacity']))
if set(vehicle_types) != set(p.keys()) or set(vehicle_types) != set(w.keys()):
    raise ValueError('Mismatch in vehicle type identifiers between products.csv columns and data.')
if set(warehouses) != set(C.keys()):
    raise ValueError('Mismatch in warehouse identifiers between capacity.csv columns and data.')
m = gp.Model('NewCarSalesInNorway2')
x_vars = m.addVars(vehicle_types, warehouses, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((p[i] * x_vars[i, j] for i in vehicle_types for j in warehouses)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((w[i] * x_vars[i, j] for i in vehicle_types)) <= C[j] for j in warehouses), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')