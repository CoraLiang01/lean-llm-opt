import gurobipy as gp
import pandas as pd
import numpy as np
import re
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/resource_limits.csv'
resource_limits_df = pd.read_csv(resource_limits_path, sep=',', dtype=str, keep_default_na=False)
resource_limits_df['Resource_norm'] = resource_limits_df['Resource'].str.strip().str.casefold()
resource_limit_map = {}
for (idx, row) in resource_limits_df.iterrows():
    resource = row['Resource'].strip()
    resource_norm = row['Resource_norm']
    try:
        limit = float(row['MonthlyLimit'])
    except Exception as e:
        raise ValueError(f"Invalid MonthlyLimit for resource '{resource}': {row['MonthlyLimit']}")
    resource_limit_map[resource_norm] = limit
required_resources = {'laborhours': 'LaborHours', 'materiala': 'MaterialA', 'materialb': 'MaterialB'}
for res_norm in required_resources:
    if res_norm not in resource_limit_map:
        raise KeyError(f"Resource '{required_resources[res_norm]}' not found in resource_limits.csv")
product_resources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/product_resources.csv'
product_resources_df = pd.read_csv(product_resources_path, sep=',', dtype=str, keep_default_na=False)
product_resources_df['Product_norm'] = product_resources_df['Product'].str.strip()
product_ids = [f'Widget{i}' for i in range(1, 142)]
labor_hours = {}
material_a = {}
material_b = {}
profit = {}
product_row_map = {row['Product'].strip(): row for (idx, row) in product_resources_df.iterrows()}
for pid in product_ids:
    if pid not in product_row_map:
        raise KeyError(f"Product '{pid}' not found in product_resources.csv")
    row = product_row_map[pid]
    try:
        labor_hours[pid] = float(row['LaborHours'])
        material_a[pid] = float(row['MaterialA'])
        material_b[pid] = float(row['MaterialB'])
        profit[pid] = float(row['Profit'])
    except Exception as e:
        raise ValueError(f"Invalid numeric value for product '{pid}': {e}")
m = gp.Model('AerospaceWidgetProduction')
x_vars = m.addVars(product_ids, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
s_var = m.addVar(lb=0.0, ub=1500.0, vtype=gp.GRB.CONTINUOUS, name='s')
d_var = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='d')
total_profit_expr = gp.quicksum((profit[pid] * x_vars[pid] for pid in product_ids)) + 300.0 * s_var - 200.0 * d_var
m.setObjective(total_profit_expr, gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((labor_hours[pid] * x_vars[pid] for pid in product_ids)) <= resource_limit_map['laborhours'], name='LaborHours')
m.addConstr(gp.quicksum((material_a[pid] * x_vars[pid] for pid in product_ids)) <= resource_limit_map['materiala'], name='MaterialA')
m.addConstr(gp.quicksum((material_b[pid] * x_vars[pid] for pid in product_ids)) <= resource_limit_map['materialb'], name='MaterialB')
if 'Widget3' not in x_vars:
    raise KeyError('Widget3 not found in product list for CatalystX byproduct constraint.')
m.addConstr(5.0 * x_vars['Widget3'] == s_var + d_var, name='CatalystX_Balance')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total profit: {m.objVal:.2f}')
    print('\n--- Production Plan ---')
    for pid in product_ids:
        qty = x_vars[pid].X
        if qty > 1e-06:
            print(f'{pid}: {qty:.2f} units')
    print('\n--- CatalystX Byproduct ---')
    print(f"Produced: {5.0 * x_vars['Widget3'].X:.2f} kg")
    print(f'Sold: {s_var.X:.2f} kg (max 1500 kg)')
    print(f'Disposed: {d_var.X:.2f} kg')
else:
    print(f'No optimal solution found. Status: {m.status}')