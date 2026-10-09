import gurobipy as gp
import pandas as pd
import numpy as np
import re
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/resource_limits.csv'
product_resources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/product_resources.csv'
resource_limits_df = pd.read_csv(resource_limits_path, sep=',', dtype=str, keep_default_na=False)
resource_limits_df['Resource_norm'] = resource_limits_df['Resource'].str.strip().str.casefold()
resource_limits_df['MonthlyLimit'] = resource_limits_df['MonthlyLimit'].astype(float)
resource_limit_map = {}
for (_, row) in resource_limits_df.iterrows():
    key = row['Resource'].strip().casefold()
    resource_limit_map[key] = row['MonthlyLimit']
product_resources_df = pd.read_csv(product_resources_path, sep=',', dtype=str, keep_default_na=False)
product_resources_df['Product'] = product_resources_df['Product'].str.strip()
product_resources_df['LaborHours'] = product_resources_df['LaborHours'].astype(float)
product_resources_df['MaterialA'] = product_resources_df['MaterialA'].astype(float)
product_resources_df['MaterialB'] = product_resources_df['MaterialB'].astype(float)
product_resources_df['Profit'] = product_resources_df['Profit'].astype(float)
widget_ids = [f'Widget{i}' for i in range(1, 142)]
missing_widgets = [wid for wid in widget_ids if wid not in set(product_resources_df['Product'])]
if missing_widgets:
    raise ValueError(f'Missing widget(s) in product_resources.csv: {missing_widgets}')
labor_hours = dict(zip(product_resources_df['Product'], product_resources_df['LaborHours']))
material_a = dict(zip(product_resources_df['Product'], product_resources_df['MaterialA']))
material_b = dict(zip(product_resources_df['Product'], product_resources_df['MaterialB']))
profit = dict(zip(product_resources_df['Product'], product_resources_df['Profit']))
m = gp.Model('WidgetProductionByproduct')
x_vars = m.addVars(widget_ids, lb=0.0, name='')
s_var = m.addVar(name='s', lb=0.0, ub=1500.0)
d_var = m.addVar(name='d', lb=0.0)
m.setObjective(gp.quicksum((profit[wid] * x_vars[wid] for wid in widget_ids)) + 300.0 * s_var - 200.0 * d_var, gp.GRB.MAXIMIZE)
labor_limit = resource_limit_map['laborhours']
m.addConstr(gp.quicksum((labor_hours[wid] * x_vars[wid] for wid in widget_ids)) <= labor_limit, name='LaborHours')
materiala_limit = resource_limit_map['materiala']
m.addConstr(gp.quicksum((material_a[wid] * x_vars[wid] for wid in widget_ids)) <= materiala_limit, name='MaterialA')
materialb_limit = resource_limit_map['materialb']
m.addConstr(gp.quicksum((material_b[wid] * x_vars[wid] for wid in widget_ids)) <= materialb_limit, name='MaterialB')
if 'Widget3' not in widget_ids:
    raise ValueError('Widget3 not found in widget_ids.')
m.addConstr(5.0 * x_vars['Widget3'] == s_var + d_var, name='CatalystX_balance')
m.optimize()