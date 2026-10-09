import gurobipy as gp
import pandas as pd
import numpy as np
import re
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/resource_limits.csv'
product_resources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/product_resources.csv'
resource_limits_df = pd.read_csv(resource_limits_path, sep=',')
product_resources_df = pd.read_csv(product_resources_path, sep=',')
widget_names = [f'Widget{i}' for i in range(1, 142)]
product_resources_df['Product'] = product_resources_df['Product'].astype(str)
missing_widgets = set(widget_names) - set(product_resources_df['Product'])
if missing_widgets:
    raise ValueError(f'Missing widget(s) in product_resources.csv: {missing_widgets}')
labor_hours = {}
material_a = {}
material_b = {}
profit = {}
for (_, row) in product_resources_df.iterrows():
    prod = str(row['Product'])
    if prod in widget_names:
        labor_hours[prod] = float(row['LaborHours'])
        material_a[prod] = float(row['MaterialA'])
        material_b[prod] = float(row['MaterialB'])
        profit[prod] = float(row['Profit'])
resource_limits_df['Resource'] = resource_limits_df['Resource'].astype(str)
resource_limits = {}
for (_, row) in resource_limits_df.iterrows():
    resource_limits[row['Resource']] = float(row['MonthlyLimit'])
labor_limit = resource_limits.get('LaborHours', None)
material_a_limit = resource_limits.get('MaterialA', None)
material_b_limit = resource_limits.get('MaterialB', None)
if labor_limit is None or material_a_limit is None or material_b_limit is None:
    raise ValueError('One or more required resource limits are missing in resource_limits.csv.')
m = gp.Model('WidgetProductionWithCatalystX')
x = m.addVars(widget_names, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
s = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='s')
d = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='d')
obj = gp.quicksum((profit[i] * x[i] for i in widget_names)) + 300 * s - 200 * d
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((labor_hours[i] * x[i] for i in widget_names)) <= labor_limit, name='LaborHoursLimit')
m.addConstr(gp.quicksum((material_a[i] * x[i] for i in widget_names)) <= material_a_limit, name='MaterialALimit')
m.addConstr(gp.quicksum((material_b[i] * x[i] for i in widget_names)) <= material_b_limit, name='MaterialBLimit')
if 'Widget3' not in widget_names:
    raise ValueError('Widget3 is missing from the widget list.')
m.addConstr(5 * x['Widget3'] == s + d, name='CatalystXBalance')
m.addConstr(s <= 1500, name='CatalystXSalesCap')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total profit: {m.objVal:.2f}')
    print('\n--- Optimal Production Plan ---')
    for i in widget_names:
        qty = x[i].X
        if qty > 1e-06:
            print(f'{i}: {qty:.2f} units')
    print(f'\nCatalystX sold: {s.X:.2f} kg (max 1500 kg)')
    print(f'CatalystX disposed: {d.X:.2f} kg')
    print(f"Total CatalystX generated: {5 * x['Widget3'].X:.2f} kg")
else:
    print(f'No optimal solution found. Status: {m.status}')