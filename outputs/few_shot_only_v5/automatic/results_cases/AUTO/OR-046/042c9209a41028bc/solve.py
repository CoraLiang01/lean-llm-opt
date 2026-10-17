import pandas as pd
import gurobipy as gp
from gurobipy import GRB
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/capacity.csv', dtype=str, keep_default_na=False)
if 'Capacity' not in capacity_df.columns or capacity_df.shape[0] == 0:
    raise ValueError('Missing or empty Capacity column in capacity.csv')
try:
    total_capacity = int(capacity_df['Capacity'].iloc[0])
except Exception as e:
    raise ValueError('Invalid capacity value in capacity.csv') from e
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv', dtype=str, keep_default_na=False)
required_columns = {'ProductName', 'Weight', 'Value'}
if not required_columns.issubset(products_df.columns):
    raise ValueError('products.csv missing required columns')
products = products_df['ProductName'].tolist()
try:
    weights = {row['ProductName']: int(row['Weight']) for _, row in products_df.iterrows()}
    values = {row['ProductName']: int(row['Value']) for _, row in products_df.iterrows()}
except Exception as e:
    raise ValueError('Invalid numeric data in products.csv') from e
m = gp.Model('SupermarketStockOrder')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((values[p] * x_vars[p] for p in products)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weights[p] * x_vars[p] for p in products)) <= total_capacity, name='stock_capacity')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in x_vars.values():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')