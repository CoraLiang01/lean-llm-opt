import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/resource_limits.csv'
product_resources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/product_resources.csv'
resource_limits_df = pd.read_csv(resource_limits_path, sep=',', dtype=str, keep_default_na=False)
product_resources_df = pd.read_csv(product_resources_path, sep=',', dtype=str, keep_default_na=False)
resource_limits_df['Resource_norm'] = resource_limits_df['Resource'].str.strip().str.casefold()
resource_limits_df['MonthlyLimit'] = resource_limits_df['MonthlyLimit'].astype(float)

def get_resource_limit(resource_name):
    norm = resource_name.strip().casefold()
    matches = resource_limits_df[resource_limits_df['Resource_norm'] == norm]
    if len(matches) != 1:
        raise ValueError(f"Resource '{resource_name}' not found or not unique in resource_limits.csv")
    return float(matches['MonthlyLimit'].iloc[0])
labor_limit = get_resource_limit('LaborHours')
materiala_limit = get_resource_limit('MaterialA')
materialb_limit = get_resource_limit('MaterialB')
product_ids = [f'Widget{i}' for i in range(1, 142)]
product_resources_df['Product_norm'] = product_resources_df['Product'].str.strip()
product_resources_df['LaborHours'] = product_resources_df['LaborHours'].astype(float)
product_resources_df['MaterialA'] = product_resources_df['MaterialA'].astype(float)
product_resources_df['MaterialB'] = product_resources_df['MaterialB'].astype(float)
product_resources_df['Profit'] = product_resources_df['Profit'].astype(float)
product_param_dict = {}
for (_, row) in product_resources_df.iterrows():
    pid = row['Product_norm']
    product_param_dict[pid] = {'LaborHours': row['LaborHours'], 'MaterialA': row['MaterialA'], 'MaterialB': row['MaterialB'], 'Profit': row['Profit']}
missing_products = [pid for pid in product_ids if pid not in product_param_dict]
if missing_products:
    raise ValueError(f'Missing product data for: {missing_products}')
m = gp.Model('widget_production')
m.Params.MIPGap = 0.0001
quantity_vars = m.addVars(product_ids, lb=0.0, vtype=GRB.CONTINUOUS, name='')
s_var = m.addVar(lb=0.0, ub=1500.0, vtype=GRB.CONTINUOUS, name='s')
d_var = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name='d')
obj = gp.quicksum((product_param_dict[i]['Profit'] * quantity_vars[i] for i in product_ids))
obj += 300.0 * s_var
obj -= 200.0 * d_var
m.setObjective(obj, GRB.MAXIMIZE)
labor_expr = gp.quicksum((product_param_dict[i]['LaborHours'] * quantity_vars[i] for i in product_ids))
m.addConstr(labor_expr <= labor_limit, name='labor')
materiala_expr = gp.quicksum((product_param_dict[i]['MaterialA'] * quantity_vars[i] for i in product_ids))
m.addConstr(materiala_expr <= materiala_limit, name='materiala')
materialb_expr = gp.quicksum((product_param_dict[i]['MaterialB'] * quantity_vars[i] for i in product_ids))
m.addConstr(materialb_expr <= materialb_limit, name='materialb')
widget3_id = 'Widget3'
m.addConstr(5.0 * quantity_vars[widget3_id] == s_var + d_var, name='catalystx_balance')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')