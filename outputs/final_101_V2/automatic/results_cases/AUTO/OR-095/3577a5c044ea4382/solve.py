import gurobipy as gp
import pandas as pd
import numpy as np
import re

def normalize_id(s):
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
labor_hours = {}
material_a = {}
material_b = {}
profit = {}
for p in products:
    row = product_resources_df.loc[p]
    labor_hours[p] = float(row['LaborHours'])
    material_a[p] = float(row['MaterialA'])
    material_b[p] = float(row['MaterialB'])
    profit[p] = float(row['Profit'])
resource_limits_df['Resource_norm'] = resource_limits_df['Resource'].apply(normalize_id)
limits_dict = dict(zip(resource_limits_df['Resource_norm'], resource_limits_df['MonthlyLimit']))
labor_limit = float(limits_dict[normalize_id('LaborHours')])
material_a_limit = float(limits_dict[normalize_id('MaterialA')])
material_b_limit = float(limits_dict[normalize_id('MaterialB')])
catalystx_byproduct_per_unit = 5.0
catalystx_sale_price = 300.0
catalystx_disposal_cost = 200.0
catalystx_sales_cap = 1500.0
m = gp.Model('AerospaceWidgetProduction')
x = m.addVars(products, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
s = m.addVar(lb=0.0, ub=catalystx_sales_cap, vtype=gp.GRB.CONTINUOUS, name='s')
d = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='d')
obj = gp.quicksum((profit[p] * x[p] for p in products)) + catalystx_sale_price * s - catalystx_disposal_cost * d
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((labor_hours[p] * x[p] for p in products)) <= labor_limit, name='LaborHours')
m.addConstr(gp.quicksum((material_a[p] * x[p] for p in products)) <= material_a_limit, name='MaterialA')
m.addConstr(gp.quicksum((material_b[p] * x[p] for p in products)) <= material_b_limit, name='MaterialB')
m.addConstr(catalystx_byproduct_per_unit * x['Widget3'] == s + d, name='CatalystX_balance')
m.addConstr(s <= catalystx_sales_cap, name='CatalystX_sales_cap')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total profit: ${m.objVal:,.2f}')
    print('\n--- Production Plan ---')
    for p in products:
        qty = x[p].X
        if qty > 1e-06:
            print(f'{p}: {qty:.2f} units')
    print('\n--- CatalystX Byproduct ---')
    print(f"Total CatalystX generated: {catalystx_byproduct_per_unit * x['Widget3'].X:.2f} kg")
    print(f'CatalystX sold: {s.X:.2f} kg (max allowed: {catalystx_sales_cap} kg)')
    print(f'CatalystX disposed: {d.X:.2f} kg')
    print(f'Revenue from CatalystX sales: ${catalystx_sale_price * s.X:,.2f}')
    print(f'Disposal cost for CatalystX: ${catalystx_disposal_cost * d.X:,.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')