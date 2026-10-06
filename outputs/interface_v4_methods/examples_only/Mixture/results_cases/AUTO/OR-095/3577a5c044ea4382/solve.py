import gurobipy as gp
import pandas as pd
import numpy as np
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/resource_limits.csv'
resource_limits_df = pd.read_csv(resource_limits_path, sep=',')
resource_limits_df['Resource_norm'] = resource_limits_df['Resource'].astype(str).str.strip().str.casefold()
resource_limits = {}
for _, row in resource_limits_df.iterrows():
    key = row['Resource'].strip().casefold()
    resource_limits[key] = float(row['MonthlyLimit'])
labor_key = 'laborhours'
materiala_key = 'materiala'
materialb_key = 'materialb'
if labor_key not in resource_limits or materiala_key not in resource_limits or materialb_key not in resource_limits:
    raise ValueError('Missing required resource limits in resource_limits.csv')
labor_limit = resource_limits[labor_key]
materiala_limit = resource_limits[materiala_key]
materialb_limit = resource_limits[materialb_key]
product_resources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/product_resources.csv'
product_df = pd.read_csv(product_resources_path, sep=',')
product_df['Product_norm'] = product_df['Product'].astype(str).str.strip()
products = [f'Widget{i}' for i in range(1, 142)]
csv_products_set = set(product_df['Product_norm'])
missing_products = [p for p in products if p not in csv_products_set]
if missing_products:
    raise ValueError(f'Missing products in product_resources.csv: {missing_products}')
labor_hours = {}
material_a = {}
material_b = {}
profit = {}
for _, row in product_df.iterrows():
    prod = row['Product'].strip()
    if prod in products:
        labor_hours[prod] = float(row['LaborHours'])
        material_a[prod] = float(row['MaterialA'])
        material_b[prod] = float(row['MaterialB'])
        profit[prod] = float(row['Profit'])
widget3 = 'Widget3'
if widget3 not in products:
    raise ValueError('Widget3 not found in product_resources.csv')
catalystx_per_unit = 5.0
catalystx_sale_price = 300.0
catalystx_disposal_cost = 200.0
catalystx_sales_cap = 1500.0
m = gp.Model('AerospaceWidgetProduction')
x = m.addVars(products, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
s = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='s')
d = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='d')
total_profit = gp.quicksum((profit[i] * x[i] for i in products)) + catalystx_sale_price * s - catalystx_disposal_cost * d
m.setObjective(total_profit, gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((labor_hours[i] * x[i] for i in products)) <= labor_limit, name='labor_limit')
m.addConstr(gp.quicksum((material_a[i] * x[i] for i in products)) <= materiala_limit, name='materiala_limit')
m.addConstr(gp.quicksum((material_b[i] * x[i] for i in products)) <= materialb_limit, name='materialb_limit')
m.addConstr(catalystx_per_unit * x[widget3] == s + d, name='catalystx_balance')
m.addConstr(s <= catalystx_sales_cap, name='catalystx_sales_cap')
m.optimize()