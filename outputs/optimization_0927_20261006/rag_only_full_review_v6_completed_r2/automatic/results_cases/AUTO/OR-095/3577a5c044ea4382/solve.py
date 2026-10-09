import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/resource_limits.csv'
resource_limits_df = pd.read_csv(resource_limits_path, dtype=str, keep_default_na=False)
resource_limits_df['Resource_norm'] = resource_limits_df['Resource'].str.strip().str.casefold()
resource_limits_df['MonthlyLimit'] = resource_limits_df['MonthlyLimit'].astype(float)
resource_limit_map = dict(zip(resource_limits_df['Resource_norm'], resource_limits_df['MonthlyLimit']))
required_resources = {'laborhours': 'LaborHours', 'materiala': 'MaterialA', 'materialb': 'MaterialB'}
for norm_name in required_resources:
    if norm_name not in resource_limit_map:
        raise ValueError(f"Resource limit for '{required_resources[norm_name]}' not found in resource_limits.csv")
labor_limit = resource_limit_map['laborhours']
materiala_limit = resource_limit_map['materiala']
materialb_limit = resource_limit_map['materialb']
product_resources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/product_resources.csv'
product_resources_df = pd.read_csv(product_resources_path, dtype=str, keep_default_na=False)
widget_ids = [f'Widget{i}' for i in range(1, 142)]
product_resources_df['Product'] = product_resources_df['Product'].str.strip()
product_resources_df = product_resources_df.set_index('Product', drop=False)
missing_widgets = [wid for wid in widget_ids if wid not in product_resources_df.index]
if missing_widgets:
    raise ValueError(f'Missing widget(s) in product_resources.csv: {missing_widgets}')
labor_hours = {}
material_a = {}
material_b = {}
profit = {}
for wid in widget_ids:
    row = product_resources_df.loc[wid]
    try:
        labor_hours[wid] = float(row['LaborHours'])
        material_a[wid] = float(row['MaterialA'])
        material_b[wid] = float(row['MaterialB'])
        profit[wid] = float(row['Profit'])
    except Exception as e:
        raise ValueError(f'Error parsing numeric fields for {wid}: {e}')
widget3_id = 'Widget3'
if widget3_id not in widget_ids:
    raise ValueError('Widget3 not found in product_resources.csv')
catalystx_byproduct_per_unit = 5.0
catalystx_sale_price = 300.0
catalystx_disposal_cost = 200.0
catalystx_sales_cap = 1500.0
m = Model('AerospaceWidgetProduction')
x_vars = m.addVars(widget_ids, lb=0.0, vtype=GRB.CONTINUOUS, name='')
s_var = m.addVar(lb=0.0, ub=catalystx_sales_cap, vtype=GRB.CONTINUOUS, name='s')
objective_expr = quicksum((profit[wid] * x_vars[wid] for wid in widget_ids)) + (catalystx_sale_price + catalystx_disposal_cost) * s_var - catalystx_disposal_cost * catalystx_byproduct_per_unit * x_vars[widget3_id]
m.setObjective(objective_expr, GRB.MAXIMIZE)
m.addConstr(quicksum((labor_hours[wid] * x_vars[wid] for wid in widget_ids)) <= labor_limit, name='LaborHoursLimit')
m.addConstr(quicksum((material_a[wid] * x_vars[wid] for wid in widget_ids)) <= materiala_limit, name='MaterialALimit')
m.addConstr(quicksum((material_b[wid] * x_vars[wid] for wid in widget_ids)) <= materialb_limit, name='MaterialBLimit')
m.addConstr(s_var >= 0.0, name='CatalystXSalesNonNeg')
m.addConstr(s_var <= catalystx_byproduct_per_unit * x_vars[widget3_id], name='CatalystXSalesProduction')
m.optimize()