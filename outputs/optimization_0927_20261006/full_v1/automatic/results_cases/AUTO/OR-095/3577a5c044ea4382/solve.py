import gurobipy as gp
import pandas as pd
import numpy as np
import re
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/resource_limits.csv'
resource_limits_df = pd.read_csv(resource_limits_path, sep=',', dtype=str, keep_default_na=False)
resource_limits_df['Resource_norm'] = resource_limits_df['Resource'].str.strip().str.casefold()
resource_limits_df['MonthlyLimit'] = resource_limits_df['MonthlyLimit'].astype(float)
resource_limit_map = {}
for (_, row) in resource_limits_df.iterrows():
    key = row['Resource'].strip().casefold()
    resource_limit_map[key] = row['MonthlyLimit']
required_resources = {'laborhours': 'LaborHours', 'materiala': 'MaterialA', 'materialb': 'MaterialB'}
for res_norm in required_resources:
    if res_norm not in resource_limit_map:
        raise ValueError(f"Resource '{required_resources[res_norm]}' not found in resource_limits.csv")
product_resources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/product_resources.csv'
product_resources_df = pd.read_csv(product_resources_path, sep=',', dtype=str, keep_default_na=False)
product_resources_df['Product_norm'] = product_resources_df['Product'].str.strip()
widget_names = [f'Widget{i}' for i in range(1, 142)]
product_set = set(product_resources_df['Product_norm'])
missing_widgets = [w for w in widget_names if w not in product_set]
if missing_widgets:
    raise ValueError(f'Missing widget(s) in product_resources.csv: {missing_widgets}')
labor_hours = {}
material_a = {}
material_b = {}
profit = {}
for (_, row) in product_resources_df.iterrows():
    prod = row['Product_norm']
    if prod in widget_names:
        try:
            labor_hours[prod] = float(row['LaborHours'])
            material_a[prod] = float(row['MaterialA'])
            material_b[prod] = float(row['MaterialB'])
            profit[prod] = float(row['Profit'])
        except Exception as e:
            raise ValueError(f"Error parsing numeric fields for product '{prod}': {e}")
m = gp.Model('WidgetProductionWithCatalystX')
x_vars = m.addVars(widget_names, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
s_var = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='s')
d_var = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='d')
obj_expr = gp.quicksum((profit[i] * x_vars[i] for i in widget_names)) + 300 * s_var - 200 * d_var
m.setObjective(obj_expr, gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((labor_hours[i] * x_vars[i] for i in widget_names)) <= resource_limit_map['laborhours'], name='LaborHoursLimit')
m.addConstr(gp.quicksum((material_a[i] * x_vars[i] for i in widget_names)) <= resource_limit_map['materiala'], name='MaterialALimit')
m.addConstr(gp.quicksum((material_b[i] * x_vars[i] for i in widget_names)) <= resource_limit_map['materialb'], name='MaterialBLimit')
m.addConstr(5.0 * x_vars['Widget3'] == s_var + d_var, name='CatalystXBalance')
m.addConstr(s_var <= 1500.0, name='CatalystXSalesCap')
m.optimize()