import pandas as pd
import gurobipy as gp
from gurobipy import GRB
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture2/41.csv'
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
required_cols = ['Product Name', 'Labor per unit', 'Material per unit', 'Selling Price', 'Variable Cost']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f'Missing required column: {col}')
products = df['Product Name'].tolist()
labor_per_unit = {row['Product Name']: float(row['Labor per unit']) for _, row in df.iterrows()}
material_per_unit = {row['Product Name']: float(row['Material per unit']) for _, row in df.iterrows()}
selling_price = {row['Product Name']: float(row['Selling Price']) for _, row in df.iterrows()}
variable_cost = {row['Product Name']: float(row['Variable Cost']) for _, row in df.iterrows()}
m = gp.Model('RedBeanClothingFactory')
x_vars = m.addVars(products, lb=0, vtype=GRB.CONTINUOUS, name='')
fixed_cost = 4500
m.setObjective(gp.quicksum(((selling_price[p] - variable_cost[p]) * x_vars[p] for p in products)) - fixed_cost, GRB.MAXIMIZE)
m.addConstr(gp.quicksum((labor_per_unit[p] * x_vars[p] for p in products)) <= 1650, name='labor_capacity')
m.addConstr(gp.quicksum((material_per_unit[p] * x_vars[p] for p in products)) <= 1850, name='material_capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')