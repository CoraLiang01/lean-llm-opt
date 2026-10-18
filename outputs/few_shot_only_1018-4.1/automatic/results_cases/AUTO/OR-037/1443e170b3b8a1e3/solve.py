import pandas as pd
import gurobipy as gp
from gurobipy import GRB
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/capacity.csv', dtype=str, keep_default_na=False)
if 'Capacity' not in capacity_df.columns or capacity_df.shape[0] == 0:
    raise ValueError('Capacity CSV missing required data.')
try:
    C = int(capacity_df['Capacity'].iloc[0])
except Exception as e:
    raise ValueError('Invalid capacity value.') from e
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/products.csv', dtype=str, keep_default_na=False)
required_cols = {'ProductName', 'Value', 'Weight'}
if not required_cols.issubset(products_df.columns):
    raise ValueError('Products CSV missing required columns.')
product_names = products_df['ProductName'].tolist()
try:
    profit = {row['ProductName']: int(row['Value']) for _, row in products_df.iterrows()}
    weight = {row['ProductName']: int(row['Weight']) for _, row in products_df.iterrows()}
except Exception as e:
    raise ValueError('Invalid product value or weight.') from e
if set(profit.keys()) != set(product_names) or set(weight.keys()) != set(product_names):
    raise ValueError('Profit/weight keys do not match product names.')
m = gp.Model('CarSales2_Inventory')
x_vars = m.addVars(product_names, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((profit[p] * x_vars[p] for p in product_names)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[p] * x_vars[p] for p in product_names)) <= C, name='capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')