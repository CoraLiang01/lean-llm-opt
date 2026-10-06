import pandas as pd
import numpy as np
from gurobipy import Model, GRB
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/resource_limits.csv'
resource_limits_df = pd.read_csv(resource_limits_path, sep=',')

def normalize_resource(s):
    return ' '.join(str(s).strip().casefold().split())
resource_limits_df['Resource_norm'] = resource_limits_df['Resource'].apply(normalize_resource)
resource_limits = dict(zip(resource_limits_df['Resource_norm'], resource_limits_df['MonthlyLimit']))
resource_map = {'laborhours': 'laborhours', 'materiala': 'materiala', 'materialb': 'materialb'}
for r in resource_map.values():
    if r not in resource_limits:
        raise ValueError(f"Resource '{r}' not found in resource_limits.csv (normalized keys: {list(resource_limits.keys())})")
labor_limit = resource_limits[resource_map['laborhours']]
materiala_limit = resource_limits[resource_map['materiala']]
materialb_limit = resource_limits[resource_map['materialb']]
product_resources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/product_resources.csv'
product_df = pd.read_csv(product_resources_path, sep=',')
product_df['Product_norm'] = product_df['Product'].apply(lambda s: str(s).strip())
product_list = [f'Widget{i}' for i in range(1, 142)]
csv_products = set(product_df['Product_norm'])
missing_products = [p for p in product_list if p not in csv_products]
if missing_products:
    raise ValueError(f'Missing products in product_resources.csv: {missing_products}')
product_df_indexed = product_df.set_index('Product_norm')
labor_hours = {}
material_a = {}
material_b = {}
profit = {}
for p in product_list:
    row = product_df_indexed.loc[p]
    labor_hours[p] = float(row['LaborHours'])
    material_a[p] = int(row['MaterialA'])
    material_b[p] = int(row['MaterialB'])
    profit[p] = int(row['Profit'])
widget3_name = 'Widget3'
if widget3_name not in product_list:
    raise ValueError('Widget3 not found in product list.')
catalystx_byproduct_per_unit = 5.0
catalystx_sale_price = 300.0
catalystx_disposal_cost = 200.0
catalystx_sales_cap = 1500.0
m = Model('AerospaceWidgetProduction')
x = m.addVars(product_list, lb=0.0, vtype=GRB.CONTINUOUS, name='')
s = m.addVar(lb=0.0, ub=catalystx_sales_cap, vtype=GRB.CONTINUOUS, name='s')
d = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name='d')
widget_profit_expr = sum((profit[i] * x[i] for i in product_list))
catalystx_revenue_expr = catalystx_sale_price * s
catalystx_disposal_cost_expr = catalystx_disposal_cost * d
m.setObjective(widget_profit_expr + catalystx_revenue_expr - catalystx_disposal_cost_expr, GRB.MAXIMIZE)
m.addConstr(sum((labor_hours[i] * x[i] for i in product_list)) <= labor_limit, name='LaborHoursLimit')
m.addConstr(sum((material_a[i] * x[i] for i in product_list)) <= materiala_limit, name='MaterialALimit')
m.addConstr(sum((material_b[i] * x[i] for i in product_list)) <= materialb_limit, name='MaterialBLimit')
m.addConstr(catalystx_byproduct_per_unit * x[widget3_name] == s + d, name='CatalystXBalance')
m.optimize()