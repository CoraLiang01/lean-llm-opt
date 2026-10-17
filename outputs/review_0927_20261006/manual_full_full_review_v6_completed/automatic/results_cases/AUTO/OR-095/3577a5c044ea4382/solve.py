import gurobipy as gp
import pandas as pd
import numpy as np
import re
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/resource_limits.csv'
product_resources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/product_resources.csv'
resource_limits_df = pd.read_csv(resource_limits_path, sep=',', dtype=str, keep_default_na=False)
resource_limits_df['Resource_norm'] = resource_limits_df['Resource'].str.strip().str.casefold()
resource_limits_df['MonthlyLimit'] = resource_limits_df['MonthlyLimit'].astype(float)
resource_limit_map = dict(zip(resource_limits_df['Resource_norm'], resource_limits_df['MonthlyLimit']))
expected_resources = {'laborhours': 'LaborHours', 'materiala': 'MaterialA', 'materialb': 'MaterialB'}
for res_norm in expected_resources:
    if res_norm not in resource_limit_map:
        raise ValueError(f"Resource '{expected_resources[res_norm]}' not found in resource_limits.csv.")
labor_limit = resource_limit_map['laborhours']
materiala_limit = resource_limit_map['materiala']
materialb_limit = resource_limit_map['materialb']
product_resources_df = pd.read_csv(product_resources_path, sep=',', dtype=str, keep_default_na=False)
product_resources_df['Product'] = product_resources_df['Product'].str.strip()
for col in ['LaborHours', 'MaterialA', 'MaterialB', 'Profit']:
    product_resources_df[col] = pd.to_numeric(product_resources_df[col], errors='raise')
widget_ids = [f'Widget{i}' for i in range(1, 142)]
missing_widgets = [wid for wid in widget_ids if wid not in set(product_resources_df['Product'])]
if missing_widgets:
    raise ValueError(f'Missing widget(s) in product_resources.csv: {missing_widgets}')
labor_hours = dict(zip(product_resources_df['Product'], product_resources_df['LaborHours']))
material_a = dict(zip(product_resources_df['Product'], product_resources_df['MaterialA']))
material_b = dict(zip(product_resources_df['Product'], product_resources_df['MaterialB']))
profit = dict(zip(product_resources_df['Product'], product_resources_df['Profit']))
m = gp.Model('AerospaceWidgetProduction')
x_vars = m.addVars(widget_ids, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
s = m.addVar(lb=0.0, ub=1500.0, vtype=gp.GRB.CONTINUOUS, name='s')
d = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='d')
m.setObjective(gp.quicksum((profit[wid] * x_vars[wid] for wid in widget_ids)) + 300 * s - 200 * d, gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((labor_hours[wid] * x_vars[wid] for wid in widget_ids)) <= labor_limit, name='LaborLimit')
m.addConstr(gp.quicksum((material_a[wid] * x_vars[wid] for wid in widget_ids)) <= materiala_limit, name='MaterialALimit')
m.addConstr(gp.quicksum((material_b[wid] * x_vars[wid] for wid in widget_ids)) <= materialb_limit, name='MaterialBLimit')
if 'Widget3' not in x_vars:
    raise ValueError('Widget3 not found in product_resources.csv.')
m.addConstr(5.0 * x_vars['Widget3'] == s + d, name='CatalystXBalance')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total profit: {m.objVal:.2f}')
    print('\n--- Production Plan ---')
    for wid in widget_ids:
        qty = x_vars[wid].X
        if qty > 1e-06:
            print(f'{wid}: {qty:.2f} units')
    print('\n--- CatalystX Byproduct ---')
    print(f'CatalystX sold: {s.X:.2f} kg (max 1500 kg)')
    print(f'CatalystX disposed: {d.X:.2f} kg')
else:
    print(f'No optimal solution found. Status: {m.status}')