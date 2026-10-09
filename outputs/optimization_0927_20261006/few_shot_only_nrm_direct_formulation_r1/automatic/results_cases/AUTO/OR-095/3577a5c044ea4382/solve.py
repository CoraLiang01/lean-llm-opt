import gurobipy as gp
import pandas as pd
import numpy as np
import re
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/resource_limits.csv'
product_resources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/product_resources.csv'
resource_limits_df = pd.read_csv(resource_limits_path, dtype=str, keep_default_na=False)
if not {'Resource', 'MonthlyLimit'}.issubset(resource_limits_df.columns):
    raise KeyError('resource_limits.csv missing required columns.')
resource_limits_df['Resource_norm'] = resource_limits_df['Resource'].str.strip().str.casefold()
resource_limits_df['MonthlyLimit_num'] = resource_limits_df['MonthlyLimit'].astype(float)
resource_limits_dict = dict(zip(resource_limits_df['Resource_norm'], resource_limits_df['MonthlyLimit_num']))

def get_limit(resource_name):
    key = resource_name.strip().casefold()
    if key not in resource_limits_dict:
        raise KeyError(f"Resource '{resource_name}' not found in resource_limits.csv")
    return resource_limits_dict[key]
labor_limit = get_limit('LaborHours')
materiala_limit = get_limit('MaterialA')
materialb_limit = get_limit('MaterialB')
product_resources_df = pd.read_csv(product_resources_path, dtype=str, keep_default_na=False)
required_cols = {'Product', 'LaborHours', 'MaterialA', 'MaterialB', 'Profit'}
if not required_cols.issubset(product_resources_df.columns):
    raise KeyError('product_resources.csv missing required columns.')
widget_names = [f'Widget{i}' for i in range(1, 142)]
product_resources_df = product_resources_df[product_resources_df['Product'].isin(widget_names)].copy()
if len(product_resources_df) != 141:
    raise ValueError(f'Expected 141 widgets, found {len(product_resources_df)} in product_resources.csv.')
product_resources_df.set_index('Product', inplace=True)
for col in ['LaborHours', 'MaterialA', 'MaterialB', 'Profit']:
    product_resources_df[col] = product_resources_df[col].astype(float)
labor_hours = product_resources_df['LaborHours'].to_dict()
material_a = product_resources_df['MaterialA'].to_dict()
material_b = product_resources_df['MaterialB'].to_dict()
profit = product_resources_df['Profit'].to_dict()
m = gp.Model('WidgetProduction')
x_vars = m.addVars(widget_names, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
s_var = m.addVar(lb=0.0, ub=1500.0, vtype=gp.GRB.CONTINUOUS, name='s')
d_var = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='d')
widget_profit_expr = gp.quicksum((profit[i] * x_vars[i] for i in widget_names))
catalystx_sale_expr = 300.0 * s_var
catalystx_disposal_expr = 200.0 * d_var
m.setObjective(widget_profit_expr + catalystx_sale_expr - catalystx_disposal_expr, gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((labor_hours[i] * x_vars[i] for i in widget_names)) <= labor_limit, name='LaborHours')
m.addConstr(gp.quicksum((material_a[i] * x_vars[i] for i in widget_names)) <= materiala_limit, name='MaterialA')
m.addConstr(gp.quicksum((material_b[i] * x_vars[i] for i in widget_names)) <= materialb_limit, name='MaterialB')
if 'Widget3' not in x_vars:
    raise KeyError('Widget3 not found in product_resources.csv.')
m.addConstr(s_var + d_var == 5.0 * x_vars['Widget3'], name='CatalystX_balance')
m.optimize()