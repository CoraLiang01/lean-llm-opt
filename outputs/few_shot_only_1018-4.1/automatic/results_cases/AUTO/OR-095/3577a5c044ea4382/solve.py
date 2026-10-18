import gurobipy as gp
import pandas as pd
import numpy as np
import re
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/resource_limits.csv'
product_resources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/product_resources.csv'
resource_limits_df = pd.read_csv(resource_limits_path, sep=',', dtype=str, keep_default_na=False)
resource_limits_df['Resource_norm'] = resource_limits_df['Resource'].str.strip().str.casefold()
resource_limits_df['MonthlyLimit'] = resource_limits_df['MonthlyLimit'].str.strip()
resource_limits_df['MonthlyLimit'] = resource_limits_df['MonthlyLimit'].astype(float)
resource_limits = {}
for idx, row in resource_limits_df.iterrows():
    resource_limits[row['Resource_norm']] = row['MonthlyLimit']
labor_limit = resource_limits['laborhours']
materiala_limit = resource_limits['materiala']
materialb_limit = resource_limits['materialb']
product_resources_df = pd.read_csv(product_resources_path, sep=',', dtype=str, keep_default_na=False)
product_resources_df['Product'] = product_resources_df['Product'].str.strip()
widget_names = [f'Widget{i}' for i in range(1, 142)]
widget_set = set(widget_names)
product_resources_df = product_resources_df[product_resources_df['Product'].isin(widget_names)].copy()
if len(product_resources_df) != 141:
    raise ValueError(f'Expected 141 widgets, found {len(product_resources_df)} in product_resources.csv.')
for col in ['LaborHours', 'MaterialA', 'MaterialB', 'Profit']:
    product_resources_df[col] = product_resources_df[col].str.strip().astype(float)
labor_hours = dict(zip(product_resources_df['Product'], product_resources_df['LaborHours']))
material_a = dict(zip(product_resources_df['Product'], product_resources_df['MaterialA']))
material_b = dict(zip(product_resources_df['Product'], product_resources_df['MaterialB']))
profit = dict(zip(product_resources_df['Product'], product_resources_df['Profit']))
m = gp.Model('WidgetProductionAndCatalystX')
x_vars = m.addVars(widget_names, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
s_var = m.addVar(lb=0.0, ub=1500.0, vtype=gp.GRB.CONTINUOUS, name='s')
d_var = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='d')
widget_profit_expr = gp.quicksum((profit[w] * x_vars[w] for w in widget_names))
catalystx_revenue_expr = 300.0 * s_var
catalystx_disposal_cost_expr = 200.0 * d_var
m.setObjective(widget_profit_expr + catalystx_revenue_expr - catalystx_disposal_cost_expr, gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((labor_hours[w] * x_vars[w] for w in widget_names)) <= labor_limit, name='LaborHoursLimit')
m.addConstr(gp.quicksum((material_a[w] * x_vars[w] for w in widget_names)) <= materiala_limit, name='MaterialALimit')
m.addConstr(gp.quicksum((material_b[w] * x_vars[w] for w in widget_names)) <= materialb_limit, name='MaterialBLimit')
m.addConstr(5.0 * x_vars['Widget3'] == s_var + d_var, name='CatalystXBalance')
m.optimize()