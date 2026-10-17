import pandas as pd
import gurobipy as gp
from gurobipy import GRB
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA2/CarSales1/products.csv', dtype=str, keep_default_na=False)
if not {'ProductName', 'Value', 'Weight'}.issubset(products_df.columns):
    raise ValueError('products.csv must contain columns: ProductName, Value, Weight')
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA2/CarSales1/capacity.csv', dtype=str, keep_default_na=False)
if 'Capacity' not in capacity_df.columns:
    raise ValueError('capacity.csv must contain column: Capacity')
try:
    total_capacity = int(capacity_df['Capacity'].iloc[0])
except Exception:
    raise ValueError('Could not parse Capacity from capacity.csv')
product_names = products_df['ProductName'].tolist()
if len(product_names) != 25:
    raise ValueError('Expected 25 vehicle types in products.csv')
try:
    value = {row['ProductName']: int(row['Value']) for _, row in products_df.iterrows()}
    weight = {row['ProductName']: int(row['Weight']) for _, row in products_df.iterrows()}
except Exception:
    raise ValueError('Could not parse Value or Weight as integers in products.csv')
for name in product_names:
    if name not in value or name not in weight:
        raise ValueError(f'Missing value or weight for product {name}')
m = gp.Model('CarSales_Inventory')
x_vars = m.addVars(product_names, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value[name] * x_vars[name] for name in product_names)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[name] * x_vars[name] for name in product_names)) <= total_capacity, name='capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')