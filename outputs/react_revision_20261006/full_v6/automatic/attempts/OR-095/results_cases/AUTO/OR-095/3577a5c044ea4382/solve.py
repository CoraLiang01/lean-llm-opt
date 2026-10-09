import gurobipy as gp
import pandas as pd
import numpy as np
import re

def normalize_id(s):
    return re.sub('\\s+', '', str(s)).casefold()
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/resource_limits.csv'
product_resources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/product_resources.csv'
resource_limits_df = pd.read_csv(resource_limits_path, sep=',', dtype=str, keep_default_na=False)
resource_limits_df['Resource_norm'] = resource_limits_df['Resource'].apply(normalize_id)
resource_limits_df['MonthlyLimit'] = resource_limits_df['MonthlyLimit'].astype(float)
resource_limits = {}
for (_, row) in resource_limits_df.iterrows():
    resource_limits[row['Resource_norm']] = row['MonthlyLimit']
required_resources = {'laborhours': None, 'materiala': None, 'materialb': None}
for k in required_resources:
    if k not in resource_limits:
        raise ValueError(f"Missing required resource limit for '{k}' in resource_limits.csv")
    required_resources[k] = resource_limits[k]
product_resources_df = pd.read_csv(product_resources_path, sep=',', dtype=str, keep_default_na=False)
product_resources_df['Product_norm'] = product_resources_df['Product'].apply(lambda s: str(s).strip())
product_ids = [f'Widget{i}' for i in range(1, 142)]
labor_hours = {}
material_a = {}
material_b = {}
profit = {}
product_resources_df.set_index('Product_norm', inplace=True)
for pid in product_ids:
    if pid not in product_resources_df.index:
        raise ValueError(f"Missing product data for '{pid}' in product_resources.csv")
    row = product_resources_df.loc[pid]
    try:
        labor_hours[pid] = float(row['LaborHours'])
        material_a[pid] = float(row['MaterialA'])
        material_b[pid] = float(row['MaterialB'])
        profit[pid] = float(row['Profit'])
    except Exception as e:
        raise ValueError(f"Invalid or missing numeric data for '{pid}': {e}")
m = gp.Model('AerospaceWidgetProduction')
quantity_vars = m.addVars(product_ids, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
s = m.addVar(lb=0.0, ub=1500.0, vtype=gp.GRB.CONTINUOUS, name='s')
d = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='d')
obj = gp.quicksum((profit[pid] * quantity_vars[pid] for pid in product_ids)) + 300.0 * s - 200.0 * d
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((labor_hours[pid] * quantity_vars[pid] for pid in product_ids)) <= required_resources['laborhours'], name='labor')
m.addConstr(gp.quicksum((material_a[pid] * quantity_vars[pid] for pid in product_ids)) <= required_resources['materiala'], name='matA')
m.addConstr(gp.quicksum((material_b[pid] * quantity_vars[pid] for pid in product_ids)) <= required_resources['materialb'], name='matB')
if 'Widget3' not in quantity_vars:
    raise ValueError('Widget3 not found in product list for CatalystX balance.')
m.addConstr(5.0 * quantity_vars['Widget3'] == s + d, name='catalyst_balance')
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for pid in product_ids:
        print(f'{quantity_vars[pid].VarName} {quantity_vars[pid].X}')
    print(f'{s.VarName} {s.X}')
    print(f'{d.VarName} {d.X}')
else:
    print(f'Solver status: {m.status}')