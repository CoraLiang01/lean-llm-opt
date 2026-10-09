import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/resource_limits.csv'
product_resources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/product_resources.csv'
resource_limits_df = pd.read_csv(resource_limits_path, sep=',', dtype=str, keep_default_na=False)
product_resources_df = pd.read_csv(product_resources_path, sep=',', dtype=str, keep_default_na=False)
widget_ids = [f'Widget{i}' for i in range(1, 142)]
product_resources_df['Product'] = product_resources_df['Product'].astype(str)
missing_widgets = [wid for wid in widget_ids if wid not in set(product_resources_df['Product'])]
if missing_widgets:
    raise ValueError(f'Missing widget(s) in product_resources.csv: {missing_widgets}')

def get_numeric_col(df, col, key_col, keys):
    col_map = {}
    for k in keys:
        v = df.loc[df[key_col] == k, col]
        if v.empty:
            raise ValueError(f'Missing value for {col} of {k}')
        try:
            col_map[k] = float(v.iloc[0])
        except Exception as e:
            raise ValueError(f'Non-numeric value for {col} of {k}: {v.iloc[0]}')
    return col_map
labor_hours = get_numeric_col(product_resources_df, 'LaborHours', 'Product', widget_ids)
material_a = get_numeric_col(product_resources_df, 'MaterialA', 'Product', widget_ids)
material_b = get_numeric_col(product_resources_df, 'MaterialB', 'Product', widget_ids)
profit = get_numeric_col(product_resources_df, 'Profit', 'Product', widget_ids)
resource_limits_df['Resource'] = resource_limits_df['Resource'].astype(str)
resource_limits = {}
for res in ['LaborHours', 'MaterialA', 'MaterialB']:
    v = resource_limits_df.loc[resource_limits_df['Resource'] == res, 'MonthlyLimit']
    if v.empty:
        raise ValueError(f'Missing resource limit for {res}')
    try:
        resource_limits[res] = float(v.iloc[0])
    except Exception as e:
        raise ValueError(f'Non-numeric resource limit for {res}: {v.iloc[0]}')
labor_limit = resource_limits['LaborHours']
material_a_limit = resource_limits['MaterialA']
material_b_limit = resource_limits['MaterialB']
m = gp.Model('WidgetProductionWithCatalystX')
x_vars = m.addVars(widget_ids, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
s_var = m.addVar(name='s', lb=0.0, vtype=gp.GRB.CONTINUOUS)
catalystx_produced_expr = 5.0 * x_vars['Widget3']
widget_profit_expr = gp.quicksum((profit[wid] * x_vars[wid] for wid in widget_ids))
catalystx_revenue_expr = 300.0 * s_var
catalystx_disposal_cost_expr = 200.0 * (catalystx_produced_expr - s_var)
m.setObjective(widget_profit_expr + catalystx_revenue_expr - catalystx_disposal_cost_expr, gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((labor_hours[wid] * x_vars[wid] for wid in widget_ids)) <= labor_limit, name='LaborHours')
m.addConstr(gp.quicksum((material_a[wid] * x_vars[wid] for wid in widget_ids)) <= material_a_limit, name='MaterialA')
m.addConstr(gp.quicksum((material_b[wid] * x_vars[wid] for wid in widget_ids)) <= material_b_limit, name='MaterialB')
m.addConstr(s_var <= 1500.0, name='CatalystX_SalesCap')
m.addConstr(s_var <= catalystx_produced_expr, name='CatalystX_SalesLEProduction')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total profit: {m.objVal:.2f}')
    print('\n--- Optimal Production Plan ---')
    for wid in widget_ids:
        val = x_vars[wid].X
        if val > 1e-06:
            print(f'{wid}: {val:.2f} units')
    print(f'\nCatalystX produced: {catalystx_produced_expr.getValue():.2f} kg')
    print(f'CatalystX sold: {s_var.X:.2f} kg')
    print(f'CatalystX disposed: {catalystx_produced_expr.getValue() - s_var.X:.2f} kg')
else:
    print(f'No optimal solution found. Status: {m.status}')