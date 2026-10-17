import gurobipy as gp
import pandas as pd
import numpy as np
import re
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/resource_limits.csv'
product_resources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/product_resources.csv'
resource_limits_df = pd.read_csv(resource_limits_path, sep=',', dtype=str, keep_default_na=False)
if set(resource_limits_df.columns) != set(['Resource', 'MonthlyLimit']):
    raise ValueError('resource_limits.csv must have columns: Resource, MonthlyLimit')
resource_limits_df['Resource_norm'] = resource_limits_df['Resource'].str.strip().str.casefold()
resource_limits_dict = {}
for _, row in resource_limits_df.iterrows():
    key = row['Resource'].strip()
    val = float(row['MonthlyLimit'])
    resource_limits_dict[key] = val
resource_norm_to_key = {row['Resource'].strip().casefold(): row['Resource'].strip() for _, row in resource_limits_df.iterrows()}
product_resources_df = pd.read_csv(product_resources_path, sep=',', dtype=str, keep_default_na=False)
expected_cols = ['Product', 'LaborHours', 'MaterialA', 'MaterialB', 'Profit']
if set(product_resources_df.columns) != set(expected_cols):
    raise ValueError('product_resources.csv must have columns: Product, LaborHours, MaterialA, MaterialB, Profit')
product_ids = [f'Widget{i}' for i in range(1, 142)]
product_resources_df['Product'] = product_resources_df['Product'].str.strip()
product_resources_df = product_resources_df[product_resources_df['Product'].isin(product_ids)]
if len(product_resources_df) != 141:
    raise ValueError(f'Expected 141 products (Widget1-Widget141), found {len(product_resources_df)} in product_resources.csv')
product_resources_df = product_resources_df.set_index('Product', drop=False)
for col in ['LaborHours', 'MaterialA', 'MaterialB', 'Profit']:
    product_resources_df[col] = product_resources_df[col].str.strip().astype(float)
labor_hours = product_resources_df['LaborHours'].to_dict()
material_a = product_resources_df['MaterialA'].to_dict()
material_b = product_resources_df['MaterialB'].to_dict()
profit = product_resources_df['Profit'].to_dict()
m = gp.Model('WidgetProductionAndCatalystX')
x_vars = m.addVars(product_ids, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
s_var = m.addVar(lb=0.0, ub=1500.0, vtype=gp.GRB.CONTINUOUS, name='s')
m.addConstr(gp.quicksum((labor_hours[i] * x_vars[i] for i in product_ids)) <= resource_limits_dict['LaborHours'], name='LaborHours')
m.addConstr(gp.quicksum((material_a[i] * x_vars[i] for i in product_ids)) <= resource_limits_dict['MaterialA'], name='MaterialA')
m.addConstr(gp.quicksum((material_b[i] * x_vars[i] for i in product_ids)) <= resource_limits_dict['MaterialB'], name='MaterialB')
m.addConstr(s_var <= 5.0 * x_vars['Widget3'], name='CatalystX_Sales_Production')
obj = gp.quicksum((profit[i] * x_vars[i] for i in product_ids)) + 500.0 * s_var - 1000.0 * x_vars['Widget3']
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total profit: {m.objVal:.2f}')
    print('\n--- Production Plan ---')
    for i in product_ids:
        val = x_vars[i].X
        if val > 1e-06:
            print(f'{i}: {val:.2f} units')
    print(f'\nCatalystX sold: {s_var.X:.2f} kg')
    print(f"CatalystX produced: {5.0 * x_vars['Widget3'].X:.2f} kg")
    print(f"CatalystX disposed: {max(0.0, 5.0 * x_vars['Widget3'].X - s_var.X):.2f} kg")
else:
    print(f'No optimal solution found. Status: {m.status}')