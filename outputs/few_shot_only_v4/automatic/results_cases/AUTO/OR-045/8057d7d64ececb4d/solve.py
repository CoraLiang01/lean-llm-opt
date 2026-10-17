import pandas as pd
import gurobipy as gp
from gurobipy import GRB
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/capacity.csv', dtype=str, keep_default_na=False)
if 'Capacity' not in capacity_df.columns:
    raise ValueError("Missing 'Capacity' column in capacity.csv")
try:
    total_capacity = float(capacity_df['Capacity'].iloc[0])
except Exception as e:
    raise ValueError(f'Could not parse capacity: {e}')
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv', dtype=str, keep_default_na=False)
required_cols = {'ProductName', 'Weight', 'Value'}
if not required_cols.issubset(products_df.columns):
    raise ValueError(f'products.csv missing columns: {required_cols - set(products_df.columns)}')
product_names = products_df['ProductName'].tolist()
if len(product_names) != len(set(product_names)):
    raise ValueError('Duplicate ProductName entries in products.csv')
try:
    weights = {row['ProductName']: float(row['Weight']) for _, row in products_df.iterrows()}
    values = {row['ProductName']: float(row['Value']) for _, row in products_df.iterrows()}
except Exception as e:
    raise ValueError(f'Could not parse numeric fields in products.csv: {e}')
m = gp.Model('Supermarket_Inventory_Optimization')
x_vars = m.addVars(product_names, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((values[p] * x_vars[p] for p in product_names)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weights[p] * x_vars[p] for p in product_names)) <= total_capacity, name='capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')