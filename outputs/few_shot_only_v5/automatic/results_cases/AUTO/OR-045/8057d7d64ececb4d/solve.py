import pandas as pd
import gurobipy as gp
from gurobipy import GRB
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/capacity.csv'
capacity_df = pd.read_csv(capacity_path, dtype=str, keep_default_na=False)
if 'Capacity' not in capacity_df.columns or capacity_df.shape[0] != 1:
    raise ValueError('Capacity CSV must have exactly one row and a "Capacity" column.')
try:
    total_capacity = int(capacity_df.loc[0, 'Capacity'])
except Exception:
    raise ValueError('Capacity value must be an integer.')
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv'
products_df = pd.read_csv(products_path, dtype=str, keep_default_na=False)
required_cols = {'ProductName', 'Weight', 'Value'}
if not required_cols.issubset(products_df.columns):
    raise ValueError('Products CSV must contain columns: ProductName, Weight, Value.')
if products_df.shape[0] != 10:
    raise ValueError('Products CSV must have exactly 10 rows as per the query.')
product_names = products_df['ProductName'].tolist()
try:
    weights = {row['ProductName']: int(row['Weight']) for _, row in products_df.iterrows()}
    values = {row['ProductName']: int(row['Value']) for _, row in products_df.iterrows()}
except Exception:
    raise ValueError('Weight and Value columns must be integers.')
m = gp.Model('Supermarket_Produce_Order')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(product_names, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((values[p] * x_vars[p] for p in product_names)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weights[p] * x_vars[p] for p in product_names)) <= total_capacity, name='total_weight')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')