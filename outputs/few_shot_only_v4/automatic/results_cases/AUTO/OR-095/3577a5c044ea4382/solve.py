import gurobipy as gp
import pandas as pd
import numpy as np
import re
product_resources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/product_resources.csv'
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/resource_limits.csv'
product_resources_df = pd.read_csv(product_resources_path, sep=',', dtype=str, keep_default_na=False)
required_columns = ['Product', 'LaborHours', 'MaterialA', 'MaterialB', 'Profit']
for col in required_columns:
    if col not in product_resources_df.columns:
        raise KeyError(f"Missing required column '{col}' in product_resources.csv")
widget_ids = [f'Widget{i}' for i in range(1, 142)]
product_resources_df['Product_norm'] = product_resources_df['Product'].str.strip()
widget_set = set(widget_ids)
df_widget_set = set(product_resources_df['Product_norm'])
missing_widgets = set(widget_ids) - df_widget_set
if missing_widgets:
    raise ValueError(f'Missing widget(s) in product_resources.csv: {sorted(missing_widgets)}')

def to_float(series, colname):
    try:
        return series.astype(float)
    except Exception as e:
        raise ValueError(f"Failed to convert column '{colname}' to float: {e}")
product_resources_df = product_resources_df.set_index('Product_norm', drop=False)
labor_hours_param = to_float(product_resources_df.loc[widget_ids, 'LaborHours'], 'LaborHours').to_dict()
material_a_param = to_float(product_resources_df.loc[widget_ids, 'MaterialA'], 'MaterialA').to_dict()
material_b_param = to_float(product_resources_df.loc[widget_ids, 'MaterialB'], 'MaterialB').to_dict()
profit_param = to_float(product_resources_df.loc[widget_ids, 'Profit'], 'Profit').to_dict()
resource_limits_df = pd.read_csv(resource_limits_path, sep=',', dtype=str, keep_default_na=False)
if 'Resource' not in resource_limits_df.columns or 'MonthlyLimit' not in resource_limits_df.columns:
    raise KeyError('Missing required columns in resource_limits.csv')
resource_limits_df['Resource_norm'] = resource_limits_df['Resource'].str.strip()
resource_limits_df = resource_limits_df.set_index('Resource_norm', drop=False)

def get_limit(resource_name):
    if resource_name not in resource_limits_df.index:
        raise ValueError(f"Resource '{resource_name}' not found in resource_limits.csv")
    try:
        return float(resource_limits_df.loc[resource_name, 'MonthlyLimit'])
    except Exception as e:
        raise ValueError(f"Failed to convert MonthlyLimit for resource '{resource_name}': {e}")
labor_hours_limit = get_limit('LaborHours')
material_a_limit = get_limit('MaterialA')
material_b_limit = get_limit('MaterialB')
m = gp.Model('AerospaceWidgetProduction')
x_vars = m.addVars(widget_ids, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
s_var = m.addVar(lb=0.0, ub=1500.0, vtype=gp.GRB.CONTINUOUS, name='s')
d_var = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='d')
total_base_profit = gp.quicksum((profit_param[i] * x_vars[i] for i in widget_ids))
catalystx_revenue = 300.0 * s_var
catalystx_disposal_cost = 200.0 * d_var
m.setObjective(total_base_profit + catalystx_revenue - catalystx_disposal_cost, gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((labor_hours_param[i] * x_vars[i] for i in widget_ids)) <= labor_hours_limit, name='LaborHoursLimit')
m.addConstr(gp.quicksum((material_a_param[i] * x_vars[i] for i in widget_ids)) <= material_a_limit, name='MaterialALimit')
m.addConstr(gp.quicksum((material_b_param[i] * x_vars[i] for i in widget_ids)) <= material_b_limit, name='MaterialBLimit')
m.addConstr(5.0 * x_vars['Widget3'] == s_var + d_var, name='CatalystXBalance')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total profit: {m.objVal:.2f}')
    print('\n--- Optimal Production Plan ---')
    for i in widget_ids:
        qty = x_vars[i].X
        if qty > 1e-06:
            print(f'{i}: {qty:.2f} units')
    print(f'\nCatalystX sold: {s_var.X:.2f} kg (max 1500 kg)')
    print(f'CatalystX disposed: {d_var.X:.2f} kg')
    print(f"Total CatalystX generated: {5.0 * x_vars['Widget3'].X:.2f} kg")
else:
    print(f'No optimal solution found. Status: {m.status}')