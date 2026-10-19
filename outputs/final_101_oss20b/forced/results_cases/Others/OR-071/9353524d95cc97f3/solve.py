import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
data_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/Mixture_testing/Mixture2/41.csv'
df = pd.read_csv(data_path, sep=',')
df.set_index('Product Name', inplace=True)
products = df.index.tolist()
m = gp.Model('RedBeanClothingProfit')
x = m.addVars(products)
labor_per_unit = df['Labor per unit']
material_per_unit = df['Material per unit']
profit_per_unit = df['Selling Price'] - df['Variable Cost']
fixed_cost = 4500
objective_expr = gp.quicksum((profit_per_unit[p] * x[p] for p in products)) - fixed_cost
m.setObjective(objective_expr, gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((labor_per_unit[p] * x[p] for p in products)) <= 1650)
m.addConstr(gp.quicksum((material_per_unit[p] * x[p] for p in products)) <= 1850)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal net profit: ${m.objVal:.2f}')
    print('\nProduction plan (units):')
    for p in products:
        qty = x[p].X
        if qty > 1e-06:
            print(f'  {p}: {qty:.2f}')
    total_labor_used = sum((labor_per_unit[p] * x[p].X for p in products))
    total_material_used = sum((material_per_unit[p] * x[p].X for p in products))
    print(f'\nTotal labor used: {total_labor_used:.2f} / 1650')
    print(f'Total material used: {total_material_used:.2f} / 1850')
else:
    print(f'Optimization did not find an optimal solution. Status: {m.status}')