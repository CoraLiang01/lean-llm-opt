import pandas as pd
import gurobipy as gp
from gurobipy import GRB
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/capacity.csv', dtype=str, keep_default_na=False)
if 'Capacity' not in capacity_df.columns or capacity_df.shape[0] == 0:
    raise ValueError('Missing or empty Capacity column in capacity.csv')
try:
    total_capacity = int(capacity_df['Capacity'].iloc[0])
except Exception as e:
    raise ValueError('Invalid capacity value in capacity.csv') from e
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', dtype=str, keep_default_na=False)
required_cols = {'ProductName', 'Value', 'Weight'}
if not required_cols.issubset(products_df.columns):
    raise ValueError('products.csv missing required columns')
products_df = products_df.reset_index(drop=True)
product_names = products_df['ProductName'].tolist()
try:
    value_dict = {row['ProductName']: int(row['Value']) for _, row in products_df.iterrows()}
    weight_dict = {row['ProductName']: int(row['Weight']) for _, row in products_df.iterrows()}
except Exception as e:
    raise ValueError('Invalid Value or Weight in products.csv') from e
m = gp.Model('Bakery_Bread_Order')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(product_names, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value_dict[p] * x_vars[p] for p in product_names)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_dict[p] * x_vars[p] for p in product_names)) <= total_capacity, name='storage')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')