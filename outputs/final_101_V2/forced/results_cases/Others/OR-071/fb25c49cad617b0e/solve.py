import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture2/41.csv'
df = pd.read_csv(csv_path, sep=',')
expected_cols = ['Product Name', 'Labor per unit', 'Material per unit', 'Selling Price', 'Variable Cost']
for col in expected_cols:
    if col not in df.columns:
        raise KeyError(f"Required column '{col}' not found in CSV.")
products = df['Product Name'].astype(str).tolist()
labor_per_unit = pd.Series(df['Labor per unit'].values, index=df['Product Name'].astype(str)).to_dict()
material_per_unit = pd.Series(df['Material per unit'].values, index=df['Product Name'].astype(str)).to_dict()
selling_price = pd.Series(df['Selling Price'].values, index=df['Product Name'].astype(str)).to_dict()
variable_cost = pd.Series(df['Variable Cost'].values, index=df['Product Name'].astype(str)).to_dict()
for pname in products:
    for d, label in [(labor_per_unit, 'Labor per unit'), (material_per_unit, 'Material per unit'), (selling_price, 'Selling Price'), (variable_cost, 'Variable Cost')]:
        if pname not in d or pd.isnull(d[pname]):
            raise ValueError(f"Missing value for '{label}' for product '{pname}'.")
labor_capacity = 1650.0
material_capacity = 1850.0
fixed_weekly_cost = 4500.0
m = gp.Model('RedBeanClothingProduction')
x = m.addVars(products, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
profit_coeff = {i: float(selling_price[i]) - float(variable_cost[i]) for i in products}
objective_expr = gp.quicksum((profit_coeff[i] * x[i] for i in products)) - fixed_weekly_cost
m.setObjective(objective_expr, gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((float(labor_per_unit[i]) * x[i] for i in products)) <= labor_capacity, name='labor_capacity')
m.addConstr(gp.quicksum((float(material_per_unit[i]) * x[i] for i in products)) <= material_capacity, name='material_capacity')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Optimal Weekly Production Plan ---')
    for i in products:
        qty = x[i].X
        if qty > 1e-06:
            print(f'{i}: {qty:.4f} units')
else:
    print(f'No optimal solution found. Status: {m.status}')