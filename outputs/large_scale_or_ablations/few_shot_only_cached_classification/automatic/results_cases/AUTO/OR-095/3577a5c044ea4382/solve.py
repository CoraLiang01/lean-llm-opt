import gurobipy as gp
import pandas as pd
import numpy as np
import re
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/resource_limits.csv'
resource_limits_df = pd.read_csv(resource_limits_path, sep=',')

def normalize_resource(s):
    return re.sub('\\s+', '', str(s)).casefold()
resource_limits_df['Resource_norm'] = resource_limits_df['Resource'].apply(normalize_resource)
resource_limits = dict(zip(resource_limits_df['Resource_norm'], resource_limits_df['MonthlyLimit']))
try:
    labor_limit = resource_limits['laborhours']
    materiala_limit = resource_limits['materiala']
    materialb_limit = resource_limits['materialb']
except KeyError as e:
    raise ValueError(f'Missing resource limit for: {e}')
product_resources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/product_resources.csv'
product_df = pd.read_csv(product_resources_path, sep=',')

def normalize_product(s):
    return str(s).strip()
product_df['Product_norm'] = product_df['Product'].apply(normalize_product)
products = [f'Widget{i}' for i in range(1, 142)]
csv_products_set = set(product_df['Product_norm'])
missing_products = [p for p in products if p not in csv_products_set]
if missing_products:
    raise ValueError(f'Missing product(s) in product_resources.csv: {missing_products}')
labor_hours = {}
material_a = {}
material_b = {}
profit = {}
for _, row in product_df.iterrows():
    prod = row['Product_norm']
    if prod in products:
        labor_hours[prod] = float(row['LaborHours'])
        material_a[prod] = float(row['MaterialA'])
        material_b[prod] = float(row['MaterialB'])
        profit[prod] = float(row['Profit'])
m = gp.Model('WidgetProductionByproduct')
x = m.addVars(products, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
s = m.addVar(lb=0.0, ub=1500.0, vtype=gp.GRB.CONTINUOUS, name='s')
d = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='d')
obj = gp.quicksum((profit[i] * x[i] for i in products)) + 300 * s - 200 * d
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((labor_hours[i] * x[i] for i in products)) <= labor_limit, name='LaborHours')
m.addConstr(gp.quicksum((material_a[i] * x[i] for i in products)) <= materiala_limit, name='MaterialA')
m.addConstr(gp.quicksum((material_b[i] * x[i] for i in products)) <= materialb_limit, name='MaterialB')
widget3 = 'Widget3'
m.addConstr(5.0 * x[widget3] == s + d, name='CatalystX_Balance')
m.addConstr(s <= 1500.0, name='CatalystX_SalesCap')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total profit: {m.objVal:.2f}')
    print('\n--- Optimal Production Plan ---')
    for i in products:
        val = x[i].X
        if val > 1e-06:
            print(f'{i}: {val:.2f} units')
    print('\n--- CatalystX Byproduct ---')
    print(f'  Sold:      {s.X:.2f} kg (max 1500 kg)')
    print(f'  Disposed:  {d.X:.2f} kg')
    print(f'  Total generated (should equal 5*Widget3): {5.0 * x[widget3].X:.2f} kg')
else:
    print(f'No optimal solution found. Status: {m.status}')