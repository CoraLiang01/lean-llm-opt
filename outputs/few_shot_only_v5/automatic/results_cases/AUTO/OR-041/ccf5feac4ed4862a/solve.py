import pandas as pd
import gurobipy as gp
from gurobipy import GRB
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/products.csv', dtype=str, keep_default_na=False)
areas = products_df['ProductName'].tolist()
try:
    values = {row['ProductName']: int(row['Value']) for _, row in products_df.iterrows()}
    weights = {row['ProductName']: int(row['Weight']) for _, row in products_df.iterrows()}
except Exception as e:
    raise ValueError(f'Error parsing products.csv: {e}')
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/capacity.csv', dtype=str, keep_default_na=False)
try:
    C = int(capacity_df.iloc[0, 0])
except Exception as e:
    raise ValueError(f'Error parsing capacity.csv: {e}')
if set(values.keys()) != set(areas) or set(weights.keys()) != set(areas):
    raise ValueError('Mismatch in area identifiers between products.csv columns and data.')
m = gp.Model('NYC_Development')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(areas, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((values[i] * x_vars[i] for i in areas)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weights[i] * x_vars[i] for i in areas)) <= C, name='capacity')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')