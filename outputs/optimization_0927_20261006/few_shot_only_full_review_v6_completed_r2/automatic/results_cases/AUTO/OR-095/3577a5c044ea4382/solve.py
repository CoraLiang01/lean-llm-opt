import gurobipy as gp
import pandas as pd
import numpy as np
import re
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/resource_limits.csv'
product_resources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/product_resources.csv'
resource_limits_df = pd.read_csv(resource_limits_path, sep=',', dtype=str, keep_default_na=False)
if not {'Resource', 'MonthlyLimit'}.issubset(resource_limits_df.columns):
    raise KeyError('resource_limits.csv missing required columns.')
resource_limits_df['Resource_norm'] = resource_limits_df['Resource'].str.strip().str.casefold()
resource_limits_dict = {}
for (idx, row) in resource_limits_df.iterrows():
    resource = row['Resource_norm']
    try:
        limit = float(row['MonthlyLimit'])
    except Exception as e:
        raise ValueError(f"Invalid MonthlyLimit for resource '{row['Resource']}': {row['MonthlyLimit']}")
    resource_limits_dict[resource] = limit
product_resources_df = pd.read_csv(product_resources_path, sep=',', dtype=str, keep_default_na=False)
required_cols = {'Product', 'LaborHours', 'MaterialA', 'MaterialB', 'Profit'}
if not required_cols.issubset(product_resources_df.columns):
    raise KeyError('product_resources.csv missing required columns.')
product_resources_df['Product_norm'] = product_resources_df['Product'].str.strip()
products = product_resources_df['Product_norm'].tolist()

def to_float_series(series, colname):
    try:
        return series.astype(float)
    except Exception as e:
        raise ValueError(f"Column '{colname}' contains non-numeric values.")
labor_hours_dict = dict(zip(product_resources_df['Product_norm'], to_float_series(product_resources_df['LaborHours'], 'LaborHours')))
material_a_dict = dict(zip(product_resources_df['Product_norm'], to_float_series(product_resources_df['MaterialA'], 'MaterialA')))
material_b_dict = dict(zip(product_resources_df['Product_norm'], to_float_series(product_resources_df['MaterialB'], 'MaterialB')))
profit_dict = dict(zip(product_resources_df['Product_norm'], to_float_series(product_resources_df['Profit'], 'Profit')))
widget3_name = None
for prod in products:
    if prod.strip().casefold() == 'widget3':
        widget3_name = prod
        break
if widget3_name is None:
    raise KeyError('Widget3 not found in product_resources.csv.')
m = gp.Model('AerospaceWidgetProduction')
x_vars = m.addVars(products, lb=0.0, name='')
s_var = m.addVar(name='s', lb=0.0, ub=1500.0)
d_var = m.addVar(name='d', lb=0.0)
labor_limit = resource_limits_dict.get('laborhours', None)
if labor_limit is None:
    raise KeyError('LaborHours resource limit not found in resource_limits.csv.')
m.addConstr(gp.quicksum((labor_hours_dict[i] * x_vars[i] for i in products)) <= labor_limit, name='LaborHours')
material_a_limit = resource_limits_dict.get('materiala', None)
if material_a_limit is None:
    raise KeyError('MaterialA resource limit not found in resource_limits.csv.')
m.addConstr(gp.quicksum((material_a_dict[i] * x_vars[i] for i in products)) <= material_a_limit, name='MaterialA')
material_b_limit = resource_limits_dict.get('materialb', None)
if material_b_limit is None:
    raise KeyError('MaterialB resource limit not found in resource_limits.csv.')
m.addConstr(gp.quicksum((material_b_dict[i] * x_vars[i] for i in products)) <= material_b_limit, name='MaterialB')
m.addConstr(5.0 * x_vars[widget3_name] == s_var + d_var, name='CatalystX_Balance')
m.setObjective(gp.quicksum((profit_dict[i] * x_vars[i] for i in products)) + 300.0 * s_var - 200.0 * d_var, gp.GRB.MAXIMIZE)
m.optimize()