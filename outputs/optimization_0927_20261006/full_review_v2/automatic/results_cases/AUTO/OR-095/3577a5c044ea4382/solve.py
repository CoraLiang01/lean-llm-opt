import gurobipy as gp
import pandas as pd
import numpy as np
import re

def norm_str(s):
    return str(s).strip().casefold()
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/resource_limits.csv'
resource_limits_df = pd.read_csv(resource_limits_path, dtype=str, keep_default_na=False)
resource_limits_df['Resource_norm'] = resource_limits_df['Resource'].apply(norm_str)
resource_limits_df['MonthlyLimit'] = resource_limits_df['MonthlyLimit'].astype(float)
resource_limit_map = {}
for (_, row) in resource_limits_df.iterrows():
    resource_limit_map[norm_str(row['Resource'])] = float(row['MonthlyLimit'])
required_resources = {'laborhours': 'LaborHours', 'materiala': 'MaterialA', 'materialb': 'MaterialB'}
for res_norm in required_resources:
    if res_norm not in resource_limit_map:
        raise ValueError(f"Resource '{required_resources[res_norm]}' not found in resource_limits.csv")
product_resources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/product_resources.csv'
product_resources_df = pd.read_csv(product_resources_path, dtype=str, keep_default_na=False)
product_resources_df['Product_norm'] = product_resources_df['Product'].apply(norm_str)
products = [f'Widget{i}' for i in range(1, 142)]
products_norm = [norm_str(p) for p in products]
product_resources_df = product_resources_df[product_resources_df['Product_norm'].isin(products_norm)].copy()
if len(product_resources_df) != 141:
    missing = set(products_norm) - set(product_resources_df['Product_norm'])
    raise ValueError(f'Missing product data for: {sorted(missing)}')
norm_to_orig_product = dict(zip(product_resources_df['Product_norm'], product_resources_df['Product']))
labor_hours = {}
material_a = {}
material_b = {}
profit = {}
for (_, row) in product_resources_df.iterrows():
    p_norm = row['Product_norm']
    labor_hours[p_norm] = float(row['LaborHours'])
    material_a[p_norm] = float(row['MaterialA'])
    material_b[p_norm] = float(row['MaterialB'])
    profit[p_norm] = float(row['Profit'])
widget3_name = 'Widget3'
widget3_norm = norm_str(widget3_name)
if widget3_norm not in products_norm:
    raise ValueError('Widget3 not found in product_resources.csv')
m = gp.Model('AerospaceWidgetProduction')
x_vars = m.addVars(products_norm, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
s_var = m.addVar(lb=0.0, ub=1500.0, vtype=gp.GRB.CONTINUOUS, name='s')
d_var = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='d')
total_profit_expr = gp.quicksum((profit[p] * x_vars[p] for p in products_norm)) + 300.0 * s_var - 200.0 * d_var
m.setObjective(total_profit_expr, gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((labor_hours[p] * x_vars[p] for p in products_norm)) <= resource_limit_map['laborhours'], name='LaborHours')
m.addConstr(gp.quicksum((material_a[p] * x_vars[p] for p in products_norm)) <= resource_limit_map['materiala'], name='MaterialA')
m.addConstr(gp.quicksum((material_b[p] * x_vars[p] for p in products_norm)) <= resource_limit_map['materialb'], name='MaterialB')
m.addConstr(5.0 * x_vars[widget3_norm] == s_var + d_var, name='CatalystX_Balance')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total profit: {m.objVal:.2f}')
    print('\n--- Production Plan ---')
    for p_norm in products_norm:
        val = x_vars[p_norm].X
        if val > 1e-06:
            print(f'{norm_to_orig_product[p_norm]}: {val:.2f} units')
    print('\n--- CatalystX Byproduct ---')
    print(f'Sold: {s_var.X:.2f} kg (max 1500 kg)')
    print(f'Disposed: {d_var.X:.2f} kg')
    print(f'Total generated (should equal 5*Widget3): {5.0 * x_vars[widget3_norm].X:.2f} kg')
else:
    print(f'No optimal solution found. Status: {m.status}')