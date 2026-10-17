import pandas as pd
import gurobipy as gp
from gurobipy import GRB
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA6/NYCPropertySales1/products.csv', dtype=str, keep_default_na=False)
if not {'Area', 'Benefit', 'Development Unit Weight'}.issubset(products_df.columns):
    raise ValueError('products.csv must contain columns: Area, Benefit, Development Unit Weight')
areas = products_df['Area'].tolist()
benefit = {}
weight = {}
for idx, row in products_df.iterrows():
    area = row['Area']
    try:
        benefit[area] = float(row['Benefit'])
        weight[area] = float(row['Development Unit Weight'])
    except Exception as e:
        raise ValueError(f'Non-numeric value in products.csv for area {area}: {e}')
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA6/NYCPropertySales1/capacity.csv', dtype=str, keep_default_na=False)
C = None
for col in capacity_df.columns:
    for val in capacity_df[col]:
        try:
            C = float(val)
            break
        except:
            continue
    if C is not None:
        break
if C is None:
    raise ValueError('No numeric capacity found in capacity.csv')
m = gp.Model('NYC_Development')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(areas, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((benefit[a] * x_vars[a] for a in areas)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[a] * x_vars[a] for a in areas)) <= C, name='capacity')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')