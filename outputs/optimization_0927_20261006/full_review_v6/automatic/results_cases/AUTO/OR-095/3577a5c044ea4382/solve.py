import gurobipy as gp
import pandas as pd
import numpy as np
import re
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/resource_limits.csv'
resource_limits_df = pd.read_csv(resource_limits_path, sep=',', dtype=str, keep_default_na=False)
resource_limits_df['Resource_norm'] = resource_limits_df['Resource'].str.strip().str.casefold()
resource_limits_df['MonthlyLimit'] = resource_limits_df['MonthlyLimit'].astype(float)
resource_limit_map = {}
for (idx, row) in resource_limits_df.iterrows():
    resource_limit_map[row['Resource_norm']] = row['MonthlyLimit']

def get_limit(resource_name):
    key = resource_name.strip().casefold()
    if key not in resource_limit_map:
        raise KeyError(f"Resource '{resource_name}' not found in resource_limits.csv")
    return resource_limit_map[key]
labor_limit = get_limit('LaborHours')
materiala_limit = get_limit('MaterialA')
materialb_limit = get_limit('MaterialB')
product_resources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/product_resources.csv'
product_resources_df = pd.read_csv(product_resources_path, sep=',', dtype=str, keep_default_na=False)
product_resources_df['Product'] = product_resources_df['Product'].str.strip()
products = product_resources_df['Product'].tolist()
if 'Widget3' not in products:
    raise KeyError('Widget3 not found in product_resources.csv')
for col in ['LaborHours', 'MaterialA', 'MaterialB', 'Profit']:
    product_resources_df[col] = pd.to_numeric(product_resources_df[col], errors='raise')
labor_hours = dict(zip(product_resources_df['Product'], product_resources_df['LaborHours']))
material_a = dict(zip(product_resources_df['Product'], product_resources_df['MaterialA']))
material_b = dict(zip(product_resources_df['Product'], product_resources_df['MaterialB']))
profit = dict(zip(product_resources_df['Product'], product_resources_df['Profit']))
m = gp.Model('AerospaceWidgetProduction')
x_vars = m.addVars(products, lb=0.0, name='')
s = m.addVar(name='s', lb=0.0, ub=1500.0)
d = m.addVar(name='d', lb=0.0)
m.setObjective(gp.quicksum((profit[p] * x_vars[p] for p in products)) + 300 * s - 200 * d, gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((labor_hours[p] * x_vars[p] for p in products)) <= labor_limit, name='LaborLimit')
m.addConstr(gp.quicksum((material_a[p] * x_vars[p] for p in products)) <= materiala_limit, name='MaterialALimit')
m.addConstr(gp.quicksum((material_b[p] * x_vars[p] for p in products)) <= materialb_limit, name='MaterialBLimit')
m.addConstr(5.0 * x_vars['Widget3'] == s + d, name='CatalystXBalance')
m.optimize()