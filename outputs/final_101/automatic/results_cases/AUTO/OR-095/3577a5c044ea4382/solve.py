import gurobipy as gp
import pandas as pd
import numpy as np
import re

def normalize_resource(s):
    return re.sub('\\s+', '', str(s)).casefold()
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/resource_limits.csv'
product_resources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/product_resources.csv'
resource_limits_df = pd.read_csv(resource_limits_path, sep=',')
product_resources_df = pd.read_csv(product_resources_path, sep=',')
products = [f'Widget{i}' for i in range(1, 142)]
csv_products_set = set(product_resources_df['Product'].astype(str))
missing_products = [p for p in products if p not in csv_products_set]
if missing_products:
    raise ValueError(f'Missing product(s) in product_resources.csv: {missing_products}')
product_resources_df = product_resources_df.set_index('Product')
labor_hours = product_resources_df['LaborHours'].astype(float).to_dict()
material_a = product_resources_df['MaterialA'].astype(float).to_dict()
material_b = product_resources_df['MaterialB'].astype(float).to_dict()
profit = product_resources_df['Profit'].astype(float).to_dict()
resource_limits_df['Resource_norm'] = resource_limits_df['Resource'].apply(normalize_resource)
limits_map = dict(zip(resource_limits_df['Resource_norm'], resource_limits_df['MonthlyLimit']))

def get_limit(resource_name):
    key = normalize_resource(resource_name)
    if key not in limits_map:
        raise ValueError(f"Resource '{resource_name}' not found in resource_limits.csv")
    return float(limits_map[key])
labor_limit = get_limit('LaborHours')
material_a_limit = get_limit('MaterialA')
material_b_limit = get_limit('MaterialB')
catalystx_byproduct_per_unit = 5.0
catalystx_sale_price = 300.0
catalystx_disposal_cost = 200.0
catalystx_sales_cap = 1500.0
m = gp.Model('WidgetProductionWithCatalystX')
x = m.addVars(products, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
s = m.addVar(lb=0.0, ub=catalystx_sales_cap, vtype=gp.GRB.CONTINUOUS, name='s')
d = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='d')
total_profit = gp.quicksum((profit[i] * x[i] for i in products)) + catalystx_sale_price * s - catalystx_disposal_cost * d
m.setObjective(total_profit, gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((labor_hours[i] * x[i] for i in products)) <= labor_limit, name='LaborHours')
m.addConstr(gp.quicksum((material_a[i] * x[i] for i in products)) <= material_a_limit, name='MaterialA')
m.addConstr(gp.quicksum((material_b[i] * x[i] for i in products)) <= material_b_limit, name='MaterialB')
m.addConstr(s + d == catalystx_byproduct_per_unit * x['Widget3'], name='CatalystX_MassBalance')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total profit: ${m.objVal:,.2f}')
    print('\n--- Optimal Production Plan ---')
    for i in products:
        qty = x[i].X
        if qty > 1e-06:
            print(f'{i}: {qty:.2f} units')
    print(f'\nCatalystX sold: {s.X:.2f} kg (max allowed: {catalystx_sales_cap} kg)')
    print(f'CatalystX disposed: {d.X:.2f} kg')
    print(f"Total CatalystX generated: {catalystx_byproduct_per_unit * x['Widget3'].X:.2f} kg")
    print('\n--- Resource Usage ---')
    used_labor = sum((labor_hours[i] * x[i].X for i in products))
    used_a = sum((material_a[i] * x[i].X for i in products))
    used_b = sum((material_b[i] * x[i].X for i in products))
    print(f'Labor hours used: {used_labor:.2f} / {labor_limit}')
    print(f'Material A used: {used_a:.2f} / {material_a_limit}')
    print(f'Material B used: {used_b:.2f} / {material_b_limit}')
else:
    print(f'No optimal solution found. Status: {m.status}')