import gurobipy as gp
import pandas as pd
import numpy as np
import re
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/resource_limits.csv'
resource_limits_df = pd.read_csv(resource_limits_path, sep=',')
resource_limits_df['Resource_norm'] = resource_limits_df['Resource'].astype(str).str.strip().str.casefold()
resource_limits = {}
for _, row in resource_limits_df.iterrows():
    key = row['Resource'].strip().casefold()
    resource_limits[key] = float(row['MonthlyLimit'])
labor_key = None
materiala_key = None
materialb_key = None
for k in resource_limits:
    if k.replace(' ', '') == 'laborhours':
        labor_key = k
    elif k.replace(' ', '') == 'materiala':
        materiala_key = k
    elif k.replace(' ', '') == 'materialb':
        materialb_key = k
if labor_key is None or materiala_key is None or materialb_key is None:
    raise ValueError('Could not find all required resource limits (LaborHours, MaterialA, MaterialB) in resource_limits.csv')
labor_limit = resource_limits[labor_key]
materiala_limit = resource_limits[materiala_key]
materialb_limit = resource_limits[materialb_key]
product_resources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/product_resources.csv'
product_df = pd.read_csv(product_resources_path, sep=',')
product_df['Product_norm'] = product_df['Product'].astype(str).str.strip()
products = [f'Widget{i}' for i in range(1, 142)]
product_set = set(product_df['Product_norm'])
missing_products = [p for p in products if p not in product_set]
if missing_products:
    raise ValueError(f'Missing product(s) in product_resources.csv: {missing_products}')
labor_hours = {}
material_a = {}
material_b = {}
profit = {}
for _, row in product_df.iterrows():
    pname = row['Product'].strip()
    if pname in products:
        labor_hours[pname] = float(row['LaborHours'])
        material_a[pname] = float(row['MaterialA'])
        material_b[pname] = float(row['MaterialB'])
        profit[pname] = float(row['Profit'])
catalystx_production = {p: 0.0 for p in products}
catalystx_production['Widget3'] = 5.0
catalystx_sale_price = 300.0
catalystx_disposal_cost = 200.0
catalystx_sales_cap = 1500.0
m = gp.Model('WidgetProduction')
x = m.addVars(products, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
s = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='s')
d = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='d')
total_profit = gp.quicksum((profit[p] * x[p] for p in products)) + catalystx_sale_price * s - catalystx_disposal_cost * d
m.setObjective(total_profit, gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((labor_hours[p] * x[p] for p in products)) <= labor_limit, name='labor')
m.addConstr(gp.quicksum((material_a[p] * x[p] for p in products)) <= materiala_limit, name='materialA')
m.addConstr(gp.quicksum((material_b[p] * x[p] for p in products)) <= materialb_limit, name='materialB')
m.addConstr(catalystx_production['Widget3'] * x['Widget3'] == s + d, name='catalystx_balance')
m.addConstr(s <= catalystx_sales_cap, name='catalystx_sales_cap')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total profit: {m.objVal:.2f}')
    print('\n--- Production Plan ---')
    for p in products:
        val = x[p].X
        if val > 1e-06:
            print(f'{p}: {val:.2f} units')
    print('\n--- CatalystX Handling ---')
    print(f'CatalystX sold: {s.X:.2f} kg')
    print(f'CatalystX disposed: {d.X:.2f} kg')
else:
    print(f'No optimal solution found. Status: {m.status}')