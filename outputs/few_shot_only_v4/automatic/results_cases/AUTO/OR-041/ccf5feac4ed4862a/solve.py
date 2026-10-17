import pandas as pd
import gurobipy as gp
from gurobipy import GRB
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/capacity.csv', dtype=str, keep_default_na=False)
if capacity_df.shape[0] == 0 or 'Capacity' not in capacity_df.columns:
    raise ValueError("Missing or malformed capacity.csv: must have at least one row and a 'Capacity' column.")
try:
    C = float(capacity_df.iloc[0]['Capacity'])
except Exception:
    raise ValueError('Capacity value in capacity.csv is not a valid number.')
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/products.csv', dtype=str, keep_default_na=False)
required_cols = {'Area', 'Value', 'Weight'}
if not required_cols.issubset(products_df.columns):
    raise ValueError(f'products.csv must contain columns: {required_cols}')
areas = products_df['Area'].tolist()
v = {}
w = {}
for idx, row in products_df.iterrows():
    area = row['Area']
    try:
        v[area] = float(row['Value'])
        w[area] = float(row['Weight'])
    except Exception:
        raise ValueError(f"Non-numeric Value or Weight for area '{area}' in products.csv.")
m = gp.Model('NYC_Development')
x_vars = m.addVars(areas, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((v[area] * x_vars[area] for area in areas)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((w[area] * x_vars[area] for area in areas)) <= C, name='capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')