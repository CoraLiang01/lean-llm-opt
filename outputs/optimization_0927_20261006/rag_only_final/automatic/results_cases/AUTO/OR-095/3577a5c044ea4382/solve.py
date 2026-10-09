import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/resource_limits.csv'
resource_limits_df = pd.read_csv(resource_limits_path, sep=',', dtype=str, keep_default_na=False)
resource_limits_df['Resource_norm'] = resource_limits_df['Resource'].str.strip().str.casefold()
resource_limits_df['MonthlyLimit'] = resource_limits_df['MonthlyLimit'].astype(float)
resource_limits_map = {}
for (_, row) in resource_limits_df.iterrows():
    resource_limits_map[row['Resource_norm']] = row['MonthlyLimit']
required_resources = {'laborhours': 'LaborHours', 'materiala': 'MaterialA', 'materialb': 'MaterialB'}
for res_norm in required_resources:
    if res_norm not in resource_limits_map:
        raise ValueError(f"Resource '{required_resources[res_norm]}' not found in resource_limits.csv")
labor_limit = resource_limits_map['laborhours']
materiala_limit = resource_limits_map['materiala']
materialb_limit = resource_limits_map['materialb']
product_resources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/product_resources.csv'
product_resources_df = pd.read_csv(product_resources_path, sep=',', dtype=str, keep_default_na=False)
product_resources_df['Product_norm'] = product_resources_df['Product'].str.strip()
product_ids = [f'Widget{i}' for i in range(1, 142)]
labor_hours = {}
material_a = {}
material_b = {}
profit = {}
for pid in product_ids:
    row = product_resources_df.loc[product_resources_df['Product_norm'] == pid]
    if row.empty:
        raise ValueError(f"Product '{pid}' not found in product_resources.csv")
    row = row.iloc[0]
    try:
        labor_hours[pid] = float(row['LaborHours'])
        material_a[pid] = float(row['MaterialA'])
        material_b[pid] = float(row['MaterialB'])
        profit[pid] = float(row['Profit'])
    except Exception as e:
        raise ValueError(f"Error parsing numeric fields for product '{pid}': {e}")
catalystx_generation_per_widget3 = 5.0
catalystx_sale_price = 300.0
catalystx_disposal_cost = 200.0
catalystx_sales_cap = 1500.0
m = Model('AerospaceWidgetProduction')
x_vars = m.addVars(product_ids, lb=0.0, vtype=GRB.CONTINUOUS, name='')
s_var = m.addVar(lb=0.0, ub=catalystx_sales_cap, vtype=GRB.CONTINUOUS, name='s')
d_var = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name='d')
m.addConstr(quicksum((labor_hours[pid] * x_vars[pid] for pid in product_ids)) <= labor_limit, name='labor_hours')
m.addConstr(quicksum((material_a[pid] * x_vars[pid] for pid in product_ids)) <= materiala_limit, name='material_a')
m.addConstr(quicksum((material_b[pid] * x_vars[pid] for pid in product_ids)) <= materialb_limit, name='material_b')
m.addConstr(catalystx_generation_per_widget3 * x_vars['Widget3'] == s_var + d_var, name='catalystx_balance')
m.addConstr(s_var <= catalystx_sales_cap, name='catalystx_sales_cap')
objective_expr = quicksum((profit[pid] * x_vars[pid] for pid in product_ids)) + catalystx_sale_price * s_var - catalystx_disposal_cost * d_var
m.setObjective(objective_expr, GRB.MAXIMIZE)
m.optimize()