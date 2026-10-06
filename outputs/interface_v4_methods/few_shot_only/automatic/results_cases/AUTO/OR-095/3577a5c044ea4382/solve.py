import gurobipy as gp
import pandas as pd
import numpy as np
import re

def find_col(df, pattern):
    for col in df.columns:
        if re.fullmatch(pattern, col.strip(), re.IGNORECASE):
            return col
    raise KeyError(f"Could not find a column matching pattern '{pattern}'")
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/resource_limits.csv'
product_resources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/product_resources.csv'
resource_limits = pd.read_csv(resource_limits_path, sep=',')
resource_limits['Resource_norm'] = resource_limits['Resource'].astype(str).str.strip().str.casefold()
resource_limits = resource_limits.set_index('Resource_norm')

def get_limit(resource_name):
    key = resource_name.strip().casefold()
    if key not in resource_limits.index:
        raise KeyError(f"Resource '{resource_name}' not found in resource_limits.csv")
    return float(resource_limits.loc[key, 'MonthlyLimit'])
labor_limit = get_limit('LaborHours')
materialA_limit = get_limit('MaterialA')
materialB_limit = get_limit('MaterialB')
product_resources = pd.read_csv(product_resources_path, sep=',')
product_resources['Product_norm'] = product_resources['Product'].astype(str).str.strip()
product_resources = product_resources.set_index('Product_norm')
products = [f'Widget{i}' for i in range(1, 142)]
missing_products = [p for p in products if p not in product_resources.index]
if missing_products:
    raise ValueError(f'Missing product(s) in product_resources.csv: {missing_products}')
labor_hours = {p: float(product_resources.loc[p, 'LaborHours']) for p in products}
materialA = {p: float(product_resources.loc[p, 'MaterialA']) for p in products}
materialB = {p: float(product_resources.loc[p, 'MaterialB']) for p in products}
profit = {p: float(product_resources.loc[p, 'Profit']) for p in products}
catalystx_byproduct_widget = 'Widget3'
catalystx_per_unit = 5.0
catalystx_sale_price = 300.0
catalystx_disposal_cost = 200.0
catalystx_sales_cap = 1500.0
m = gp.Model('AerospaceWidgetProduction')
x = m.addVars(products, lb=0.0, name='')
s = m.addVar(name='s', lb=0.0)
d = m.addVar(name='d', lb=0.0)
obj = gp.quicksum((profit[p] * x[p] for p in products)) + catalystx_sale_price * s - catalystx_disposal_cost * d
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((labor_hours[p] * x[p] for p in products)) <= labor_limit, name='LaborHours')
m.addConstr(gp.quicksum((materialA[p] * x[p] for p in products)) <= materialA_limit, name='MaterialA')
m.addConstr(gp.quicksum((materialB[p] * x[p] for p in products)) <= materialB_limit, name='MaterialB')
m.addConstr(catalystx_per_unit * x[catalystx_byproduct_widget] == s + d, name='CatalystX_Balance')
m.addConstr(s <= catalystx_sales_cap, name='CatalystX_SalesCap')
m.optimize()