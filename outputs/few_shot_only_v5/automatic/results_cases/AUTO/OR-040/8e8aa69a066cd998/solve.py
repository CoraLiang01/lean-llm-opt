import pandas as pd
import gurobipy as gp
from gurobipy import GRB
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA6/NYCPropertySales1/capacity.csv', dtype=str, keep_default_na=False)
if 'Capacity' not in capacity_df.columns or capacity_df.shape[0] == 0:
    raise ValueError('Missing or empty Capacity column in capacity.csv')
try:
    total_capacity = int(capacity_df['Capacity'].iloc[0])
except Exception as e:
    raise ValueError('Invalid capacity value in capacity.csv') from e
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA6/NYCPropertySales1/products.csv', dtype=str, keep_default_na=False)
required_cols = {'ProductName', 'Value', 'Weight'}
if not required_cols.issubset(products_df.columns):
    raise ValueError('products.csv missing required columns')
areas = products_df['ProductName'].tolist()
if len(areas) != 20:
    raise ValueError('Expected 20 areas in products.csv')
try:
    benefit = {row['ProductName']: int(row['Value']) for _, row in products_df.iterrows()}
    weight = {row['ProductName']: int(row['Weight']) for _, row in products_df.iterrows()}
except Exception as e:
    raise ValueError('Invalid Value or Weight in products.csv') from e
m = gp.Model('NYC_Development')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(areas, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((benefit[a] * x_vars[a] for a in areas)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[a] * x_vars[a] for a in areas)) <= total_capacity, name='capacity')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')