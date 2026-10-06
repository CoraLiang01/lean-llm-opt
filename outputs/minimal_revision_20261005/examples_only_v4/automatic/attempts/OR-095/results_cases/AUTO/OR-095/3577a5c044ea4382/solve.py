import gurobipy as gp
import pandas as pd
import numpy as np
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/resource_limits.csv'
product_resources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/product_resources.csv'
resource_limits_df = pd.read_csv(resource_limits_path, sep=',')
resource_limits_df['Resource_norm'] = resource_limits_df['Resource'].astype(str).str.strip().str.casefold()
resource_limits = {}
for (_, row) in resource_limits_df.iterrows():
    resource_limits[row['Resource_norm']] = float(row['MonthlyLimit'])
product_resources_df = pd.read_csv(product_resources_path, sep=',')
product_resources_df['Product_norm'] = product_resources_df['Product'].astype(str).str.strip()
products = list(product_resources_df['Product_norm'])
expected_products = [f'Widget{i}' for i in range(1, 142)]
missing = set(expected_products) - set(products)
if missing:
    raise ValueError(f'Missing products in product_resources.csv: {missing}')
if len(set(products)) != 141:
    raise ValueError('Duplicate or missing product identifiers in product_resources.csv.')
labor_hours = dict(zip(product_resources_df['Product_norm'], product_resources_df['LaborHours']))
material_a = dict(zip(product_resources_df['Product_norm'], product_resources_df['MaterialA']))
material_b = dict(zip(product_resources_df['Product_norm'], product_resources_df['MaterialB']))
profit = dict(zip(product_resources_df['Product_norm'], product_resources_df['Profit']))
labor_limit = resource_limits['laborhours']
materiala_limit = resource_limits['materiala']
materialb_limit = resource_limits['materialb']
catalystx_byproduct_per_unit = 5.0
catalystx_sale_price = 300.0
catalystx_disposal_cost = 200.0
catalystx_sales_cap = 1500.0
m = gp.Model('AerospaceWidgetProduction')
x = m.addVars(products, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
s = m.addVar(lb=0.0, ub=catalystx_sales_cap, vtype=gp.GRB.CONTINUOUS, name='s')
d = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='d')
obj = gp.quicksum((profit[i] * x[i] for i in products)) + catalystx_sale_price * s - catalystx_disposal_cost * d
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((labor_hours[i] * x[i] for i in products)) <= labor_limit, name='labor')
m.addConstr(gp.quicksum((material_a[i] * x[i] for i in products)) <= materiala_limit, name='matA')
m.addConstr(gp.quicksum((material_b[i] * x[i] for i in products)) <= materialb_limit, name='matB')
widget3_key = 'Widget3'
if widget3_key not in products:
    raise ValueError('Widget3 not found in product_resources.csv.')
m.addConstr(catalystx_byproduct_per_unit * x[widget3_key] == s + d, name='catalystx_balance')
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for i in products:
        print(f'{x[i].VarName} {x[i].X}')
    print(f'{s.VarName} {s.X}')
    print(f'{d.VarName} {d.X}')
else:
    print(f'Solver status: {m.status}')