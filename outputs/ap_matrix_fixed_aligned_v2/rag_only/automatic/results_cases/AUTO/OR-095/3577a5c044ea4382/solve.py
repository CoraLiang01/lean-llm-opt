import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/resource_limits.csv'
resource_limits_df = pd.read_csv(resource_limits_path, dtype=str, keep_default_na=False)
resource_limits_df['Resource_norm'] = resource_limits_df['Resource'].str.strip().str.casefold()
resource_limits_df['MonthlyLimit'] = resource_limits_df['MonthlyLimit'].astype(float)
resource_name_map = {'laborhours': 'LaborHours', 'materiala': 'MaterialA', 'materialb': 'MaterialB'}
resource_limits = {}
for norm_name, orig_name in resource_name_map.items():
    match = resource_limits_df['Resource_norm'] == norm_name
    if not match.any():
        raise ValueError(f"Resource '{orig_name}' not found in resource_limits.csv")
    limit = resource_limits_df.loc[match, 'MonthlyLimit']
    if len(limit) != 1:
        raise ValueError(f"Resource '{orig_name}' matched {len(limit)} rows in resource_limits.csv")
    resource_limits[orig_name] = float(limit.iloc[0])
product_resources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/product_resources.csv'
product_resources_df = pd.read_csv(product_resources_path, dtype=str, keep_default_na=False)
product_resources_df['Product'] = product_resources_df['Product'].str.strip()
product_ids = [f'Widget{i}' for i in range(1, 142)]
products_in_file = set(product_resources_df['Product'])
missing_products = set(product_ids) - products_in_file
if missing_products:
    raise ValueError(f'Missing products in product_resources.csv: {missing_products}')
product_resources_df = product_resources_df.set_index('Product', drop=False)
for col in ['LaborHours', 'MaterialA', 'MaterialB', 'Profit']:
    product_resources_df[col] = pd.to_numeric(product_resources_df[col], errors='raise')
labor_hours = product_resources_df['LaborHours'].to_dict()
material_a = product_resources_df['MaterialA'].to_dict()
material_b = product_resources_df['MaterialB'].to_dict()
profit = product_resources_df['Profit'].to_dict()
widget3_id = 'Widget3'
if widget3_id not in product_ids:
    raise ValueError('Widget3 not found in product_resources.csv')
m = Model('AerospaceWidgetProduction')
x_vars = m.addVars(product_ids, lb=0.0, vtype=GRB.CONTINUOUS, name='')
s_var = m.addVar(lb=0.0, ub=1500.0, vtype=GRB.CONTINUOUS, name='s')
d_var = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name='d')
obj = quicksum((profit[i] * x_vars[i] for i in product_ids)) + 300.0 * s_var - 200.0 * d_var
m.setObjective(obj, GRB.MAXIMIZE)
m.addConstr(quicksum((labor_hours[i] * x_vars[i] for i in product_ids)) <= resource_limits['LaborHours'], name='LaborHoursLimit')
m.addConstr(quicksum((material_a[i] * x_vars[i] for i in product_ids)) <= resource_limits['MaterialA'], name='MaterialALimit')
m.addConstr(quicksum((material_b[i] * x_vars[i] for i in product_ids)) <= resource_limits['MaterialB'], name='MaterialBLimit')
m.addConstr(5.0 * x_vars[widget3_id] == s_var + d_var, name='CatalystXBalance')
m.optimize()