import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/resource_limits.csv'
resource_limits_df = pd.read_csv(resource_limits_path, sep=',')

def normalize_resource(s):
    return ' '.join(str(s).strip().split()).casefold()
resource_limits_df['Resource_norm'] = resource_limits_df['Resource'].apply(normalize_resource)
resource_limits = dict(zip(resource_limits_df['Resource_norm'], resource_limits_df['MonthlyLimit']))
required_resources = {'laborhours': None, 'materiala': None, 'materialb': None}
for k in required_resources:
    if k not in resource_limits:
        raise ValueError(f"Resource '{k}' not found in resource_limits.csv (normalized keys: {list(resource_limits.keys())})")
    required_resources[k] = resource_limits[k]
product_resources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/product_resources.csv'
product_df = pd.read_csv(product_resources_path, sep=',')
product_df['Product_norm'] = product_df['Product'].apply(lambda s: str(s).strip())
product_ids = [f'Widget{i}' for i in range(1, 142)]
product_df_indexed = product_df.set_index('Product_norm')
missing_products = [pid for pid in product_ids if pid not in product_df_indexed.index]
if missing_products:
    raise ValueError(f'Missing products in product_resources.csv: {missing_products}')
LaborHours = {pid: float(product_df_indexed.loc[pid, 'LaborHours']) for pid in product_ids}
MaterialA = {pid: int(product_df_indexed.loc[pid, 'MaterialA']) for pid in product_ids}
MaterialB = {pid: int(product_df_indexed.loc[pid, 'MaterialB']) for pid in product_ids}
Profit = {pid: int(product_df_indexed.loc[pid, 'Profit']) for pid in product_ids}
widget3_id = 'Widget3'
if widget3_id not in product_ids:
    raise ValueError('Widget3 not found in product_resources.csv')
catalystx_per_widget3 = 5.0
catalystx_sale_price = 300.0
catalystx_disposal_cost = 200.0
catalystx_sales_cap = 1500.0
m = gp.Model('AerospaceWidgetProduction')
x = m.addVars(product_ids, lb=0.0, vtype=GRB.CONTINUOUS, name='')
s = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name='s')
d = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name='d')
obj_expr = gp.quicksum((Profit[pid] * x[pid] for pid in product_ids)) + catalystx_sale_price * s - catalystx_disposal_cost * d
m.setObjective(obj_expr, GRB.MAXIMIZE)
m.addConstr(gp.quicksum((LaborHours[pid] * x[pid] for pid in product_ids)) <= required_resources['laborhours'], name='LaborHoursLimit')
m.addConstr(gp.quicksum((MaterialA[pid] * x[pid] for pid in product_ids)) <= required_resources['materiala'], name='MaterialALimit')
m.addConstr(gp.quicksum((MaterialB[pid] * x[pid] for pid in product_ids)) <= required_resources['materialb'], name='MaterialBLimit')
m.addConstr(catalystx_per_widget3 * x[widget3_id] == s + d, name='CatalystXBalance')
m.addConstr(s <= catalystx_sales_cap, name='CatalystXSalesCap')
m.optimize()