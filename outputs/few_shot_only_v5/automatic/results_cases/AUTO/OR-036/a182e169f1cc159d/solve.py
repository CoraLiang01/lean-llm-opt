import pandas as pd
import gurobipy as gp
from gurobipy import GRB
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA2/CarSales1/capacity.csv', dtype=str, keep_default_na=False)
if 'Capacity' not in capacity_df.columns or capacity_df['Capacity'].isnull().any():
    raise ValueError("Missing or invalid 'Capacity' in capacity.csv")
try:
    capacity = int(capacity_df['Capacity'].iloc[0])
except Exception as e:
    raise ValueError('Invalid capacity value in capacity.csv') from e
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA2/CarSales1/products.csv', dtype=str, keep_default_na=False)
required_cols = {'ProductName', 'Value', 'Weight'}
if not required_cols.issubset(products_df.columns):
    raise ValueError(f'Missing columns in products.csv: {required_cols - set(products_df.columns)}')
products_df = products_df.dropna(subset=['ProductName', 'Value', 'Weight'])
product_names = products_df['ProductName'].tolist()
if len(product_names) != len(set(product_names)):
    raise ValueError('Duplicate ProductName entries in products.csv')
try:
    value = {row['ProductName']: int(row['Value']) for _, row in products_df.iterrows()}
    weight = {row['ProductName']: int(row['Weight']) for _, row in products_df.iterrows()}
except Exception as e:
    raise ValueError('Invalid Value or Weight in products.csv') from e
for p in product_names:
    if p not in value or p not in weight:
        raise ValueError(f'Missing value or weight for product {p}')
m = gp.Model('CarSales1_Inventory')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(product_names, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value[p] * x_vars[p] for p in product_names)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[p] * x_vars[p] for p in product_names)) <= capacity, name='capacity')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')