import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
resource_limits_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/resource_limits.csv'
product_resources_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/product_resources.csv'
df_limits = pd.read_csv(resource_limits_path)
df_limits['Resource'] = df_limits['Resource'].str.strip()
limits = dict(zip(df_limits['Resource'], df_limits['MonthlyLimit']))
labor_limit = limits.get('LaborHours', 0)
material_a_limit = limits.get('MaterialA', 0)
material_b_limit = limits.get('MaterialB', 0)
df_products = pd.read_csv(product_resources_path)
df_products['Product'] = df_products['Product'].str.strip()
products = df_products['Product'].tolist()
labor_hours = dict(zip(df_products['Product'], df_products['LaborHours']))
material_a = dict(zip(df_products['Product'], df_products['MaterialA']))
material_b = dict(zip(df_products['Product'], df_products['MaterialB']))
profit = dict(zip(df_products['Product'], df_products['Profit']))
m = gp.Model('WidgetProduction')
x = m.addVars(products)
s = m.addVar(lb=0)
u = m.addVar(lb=0)
m.addConstr(gp.quicksum((labor_hours[p] * x[p] for p in products)) <= labor_limit)
m.addConstr(gp.quicksum((material_a[p] * x[p] for p in products)) <= material_a_limit)
m.addConstr(gp.quicksum((material_b[p] * x[p] for p in products)) <= material_b_limit)
m.addConstr(u == 5 * x['Widget3'] - s)
m.addConstr(s <= 1500)
objective = gp.quicksum((profit[p] * x[p] for p in products)) + 300 * s - 200 * u
m.setObjective(objective, gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal objective value: {m.objVal:.2f}')
    print('\nProduction plan (units):')
    for prod in products:
        qty = x[prod].X
        if qty > 1e-06:
            print(f'  {prod}: {qty:.2f}')
    print(f'\nCatalystX sold (kg): {s.X:.2f}')
    print(f'CatalystX disposed (kg): {u.X:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')