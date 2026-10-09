import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/resource_limits.csv'
product_resources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/product_resources.csv'
resource_limits_df = pd.read_csv(resource_limits_path, dtype=str, keep_default_na=False)
product_resources_df = pd.read_csv(product_resources_path, dtype=str, keep_default_na=False)
widget_ids = [f'Widget{i}' for i in range(1, 142)]
product_resources_df['Product'] = product_resources_df['Product'].astype(str)
missing_widgets = set(widget_ids) - set(product_resources_df['Product'])
if missing_widgets:
    raise ValueError(f'Missing widget(s) in product_resources.csv: {missing_widgets}')

def get_numeric_col(df, key_col, value_col, ids):
    sub = df[[key_col, value_col]].set_index(key_col)
    missing = set(ids) - set(sub.index)
    if missing:
        raise ValueError(f'Missing {value_col} for: {missing}')
    return {i: float(sub.loc[i, value_col]) for i in ids}
labor_hours = get_numeric_col(product_resources_df, 'Product', 'LaborHours', widget_ids)
material_a = get_numeric_col(product_resources_df, 'Product', 'MaterialA', widget_ids)
material_b = get_numeric_col(product_resources_df, 'Product', 'MaterialB', widget_ids)
profit = get_numeric_col(product_resources_df, 'Product', 'Profit', widget_ids)
resource_limits_df['Resource'] = resource_limits_df['Resource'].astype(str)
resource_limits_df['MonthlyLimit'] = resource_limits_df['MonthlyLimit'].astype(str)

def get_resource_limit(resource_name):
    match = resource_limits_df[resource_limits_df['Resource'] == resource_name]
    if match.empty:
        raise ValueError(f"Resource limit for '{resource_name}' not found in resource_limits.csv")
    return float(match.iloc[0]['MonthlyLimit'])
labor_limit = get_resource_limit('LaborHours')
material_a_limit = get_resource_limit('MaterialA')
material_b_limit = get_resource_limit('MaterialB')
catalystx_per_widget3 = 5.0
catalystx_sale_price = 300.0
catalystx_disposal_cost = 200.0
catalystx_sales_cap = 1500.0
m = gp.Model('WidgetProductionWithCatalystX')
x_vars = m.addVars(widget_ids, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
s_var = m.addVar(lb=0.0, ub=catalystx_sales_cap, vtype=gp.GRB.CONTINUOUS, name='s')
objective = gp.quicksum((profit[i] * x_vars[i] for i in widget_ids)) + 500.0 * s_var - 1000.0 * x_vars['Widget3']
m.setObjective(objective, gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((labor_hours[i] * x_vars[i] for i in widget_ids)) <= labor_limit, name='LaborHoursLimit')
m.addConstr(gp.quicksum((material_a[i] * x_vars[i] for i in widget_ids)) <= material_a_limit, name='MaterialALimit')
m.addConstr(gp.quicksum((material_b[i] * x_vars[i] for i in widget_ids)) <= material_b_limit, name='MaterialBLimit')
m.addConstr(s_var <= catalystx_per_widget3 * x_vars['Widget3'], name='CatalystXSoldLEProduced')
m.optimize()