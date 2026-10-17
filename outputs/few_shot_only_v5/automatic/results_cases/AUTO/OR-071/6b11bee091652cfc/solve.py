import pandas as pd
import gurobipy as gp
from gurobipy import GRB
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture2/41.csv'
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
products = df['Product Name'].tolist()
labor_per_unit = {row['Product Name']: float(row['Labor per unit']) for _, row in df.iterrows()}
material_per_unit = {row['Product Name']: float(row['Material per unit']) for _, row in df.iterrows()}
selling_price = {row['Product Name']: float(row['Selling Price']) for _, row in df.iterrows()}
variable_cost = {row['Product Name']: float(row['Variable Cost']) for _, row in df.iterrows()}
m = gp.Model('RedBeanClothingFactory')
x_vars = m.addVars(products, lb=0, vtype=GRB.CONTINUOUS, name='')
profit_expr = gp.quicksum(((selling_price[p] - variable_cost[p]) * x_vars[p] for p in products))
m.setObjective(profit_expr - 4500, GRB.MAXIMIZE)
m.addConstr(gp.quicksum((labor_per_unit[p] * x_vars[p] for p in products)) <= 1650, name='labor_cap')
m.addConstr(gp.quicksum((material_per_unit[p] * x_vars[p] for p in products)) <= 1850, name='material_cap')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')