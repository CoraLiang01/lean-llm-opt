import pandas as pd
import gurobipy as gp
from gurobipy import GRB
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA8/PharmaSalesData1/products.csv', dtype=str, keep_default_na=False)
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA8/PharmaSalesData1/capacity.csv', dtype=str, keep_default_na=False)
if not {'ProductName', 'Value', 'Weight'}.issubset(products_df.columns):
    raise ValueError('products.csv must contain columns: ProductName, Value, Weight')
products = list(products_df['ProductName'])
benefit = {row['ProductName']: int(row['Value']) for _, row in products_df.iterrows()}
weight = {row['ProductName']: int(row['Weight']) for _, row in products_df.iterrows()}
if not {'Resource', 'Capacity'}.issubset(capacity_df.columns):
    raise ValueError('capacity.csv must contain columns: Resource, Capacity')
total_capacity = capacity_df['Capacity'].astype(float).sum()
for p in products:
    if p not in benefit or p not in weight:
        raise ValueError(f'Missing benefit or weight for product {p}')
m = gp.Model('Pharmacy_Inventory_Optimization')
x_vars = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((benefit[p] * x_vars[p] for p in products)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[p] * x_vars[p] for p in products)) <= total_capacity, name='capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')