import gurobipy as gp
import pandas as pd
import numpy as np
import re

def normalize_str(s):
    return re.sub('\\s+', '', str(s)).casefold()
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/resource_limits.csv'
resource_limits_df = pd.read_csv(resource_limits_path, dtype=str, keep_default_na=False)
resource_limits_df['Resource_norm'] = resource_limits_df['Resource'].apply(normalize_str)
resource_limits_df['MonthlyLimit'] = resource_limits_df['MonthlyLimit'].astype(float)
resource_limits = {}
for _, row in resource_limits_df.iterrows():
    resource_limits[row['Resource_norm']] = row['MonthlyLimit']
required_resources = {'laborhours': None, 'materiala': None, 'materialb': None}
for res in required_resources:
    if res not in resource_limits:
        raise ValueError(f"Resource '{res}' not found in resource_limits.csv")
    required_resources[res] = resource_limits[res]
product_resources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/product_resources.csv'
product_resources_df = pd.read_csv(product_resources_path, dtype=str, keep_default_na=False)
product_resources_df['Product_norm'] = product_resources_df['Product'].apply(normalize_str)
product_ids = [f'Widget{i}' for i in range(1, 142)]
product_ids_norm = [normalize_str(pid) for pid in product_ids]
labor_hours = {}
material_a = {}
material_b = {}
profit = {}
product_resources_df.set_index('Product_norm', inplace=True)
for pid, pid_norm in zip(product_ids, product_ids_norm):
    if pid_norm not in product_resources_df.index:
        raise ValueError(f"Product '{pid}' not found in product_resources.csv")
    row = product_resources_df.loc[pid_norm]
    try:
        labor_hours[pid] = float(row['LaborHours'])
        material_a[pid] = float(row['MaterialA'])
        material_b[pid] = float(row['MaterialB'])
        profit[pid] = float(row['Profit'])
    except Exception as e:
        raise ValueError(f"Malformed numeric data for product '{pid}': {e}")
m = gp.Model('AerospaceWidgetProduction')
x_vars = m.addVars(product_ids, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
s_var = m.addVar(lb=0.0, ub=1500.0, vtype=gp.GRB.CONTINUOUS, name='s')
d_var = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='d')
total_profit = gp.quicksum((profit[pid] * x_vars[pid] for pid in product_ids))
total_profit += 300.0 * s_var
total_profit -= 200.0 * d_var
m.setObjective(total_profit, gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((labor_hours[pid] * x_vars[pid] for pid in product_ids)) <= required_resources['laborhours'], name='LaborHours')
m.addConstr(gp.quicksum((material_a[pid] * x_vars[pid] for pid in product_ids)) <= required_resources['materiala'], name='MaterialA')
m.addConstr(gp.quicksum((material_b[pid] * x_vars[pid] for pid in product_ids)) <= required_resources['materialb'], name='MaterialB')
if 'Widget3' not in x_vars:
    raise ValueError('Widget3 not found in product list for CatalystX byproduct constraint.')
m.addConstr(5.0 * x_vars['Widget3'] == s_var + d_var, name='CatalystX_Balance')
m.optimize()