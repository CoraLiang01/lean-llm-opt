import gurobipy as gp
import pandas as pd
import numpy as np
import re
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/resource_limits.csv'
product_resources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/product_resources.csv'
resource_limits_df = pd.read_csv(resource_limits_path, sep=',', dtype=str, keep_default_na=False)
resource_limits_df['Resource_norm'] = resource_limits_df['Resource'].str.strip().str.casefold()
resource_limits_df['MonthlyLimit'] = resource_limits_df['MonthlyLimit'].astype(float)
resource_limit_map = dict(zip(resource_limits_df['Resource_norm'], resource_limits_df['MonthlyLimit']))
labor_key = 'laborhours'
materiala_key = 'materiala'
materialb_key = 'materialb'
if labor_key not in resource_limit_map or materiala_key not in resource_limit_map or materialb_key not in resource_limit_map:
    raise ValueError('One or more required resources (LaborHours, MaterialA, MaterialB) not found in resource_limits.csv.')
labor_limit = resource_limit_map[labor_key]
materiala_limit = resource_limit_map[materiala_key]
materialb_limit = resource_limit_map[materialb_key]
product_resources_df = pd.read_csv(product_resources_path, sep=',', dtype=str, keep_default_na=False)
product_resources_df['Product_norm'] = product_resources_df['Product'].str.strip()
product_ids = [f'Widget{i}' for i in range(1, 142)]
available_products = set(product_resources_df['Product_norm'])
missing_products = [pid for pid in product_ids if pid not in available_products]
if missing_products:
    raise ValueError(f'Missing products in product_resources.csv: {missing_products}')

def get_col_numeric(df, col, idx_col, idx_list, dtype):
    col_vals = {}
    for pid in idx_list:
        row = df.loc[df[idx_col] == pid]
        if row.empty:
            raise ValueError(f'Product {pid} missing in product_resources.csv.')
        val = row.iloc[0][col]
        try:
            col_vals[pid] = dtype(val)
        except Exception as e:
            raise ValueError(f'Invalid value for {col} of {pid}: {val}')
    return col_vals
labor_hours = get_col_numeric(product_resources_df, 'LaborHours', 'Product_norm', product_ids, float)
material_a = get_col_numeric(product_resources_df, 'MaterialA', 'Product_norm', product_ids, float)
material_b = get_col_numeric(product_resources_df, 'MaterialB', 'Product_norm', product_ids, float)
profit = get_col_numeric(product_resources_df, 'Profit', 'Product_norm', product_ids, float)
m = gp.Model('WidgetProductionWithCatalystX')
x_vars = m.addVars(product_ids, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
s = m.addVar(lb=0.0, ub=1500.0, vtype=gp.GRB.CONTINUOUS, name='s')
d = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='d')
m.setObjective(gp.quicksum((profit[pid] * x_vars[pid] for pid in product_ids)) + 300.0 * s - 200.0 * d, gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((labor_hours[pid] * x_vars[pid] for pid in product_ids)) <= labor_limit, name='LaborHours')
m.addConstr(gp.quicksum((material_a[pid] * x_vars[pid] for pid in product_ids)) <= materiala_limit, name='MaterialA')
m.addConstr(gp.quicksum((material_b[pid] * x_vars[pid] for pid in product_ids)) <= materialb_limit, name='MaterialB')
widget3_id = 'Widget3'
if widget3_id not in product_ids:
    raise ValueError('Widget3 not found in product list.')
m.addConstr(s + d == 5.0 * x_vars[widget3_id], name='CatalystX_Balance')
m.optimize()