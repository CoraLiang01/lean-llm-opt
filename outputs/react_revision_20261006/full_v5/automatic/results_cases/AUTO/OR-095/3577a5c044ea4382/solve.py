import gurobipy as gp
import pandas as pd
import numpy as np
import re

def normalize_resource(s):
    return re.sub('\\s+', '', str(s)).casefold()

def normalize_product(s):
    return str(s).strip()
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/resource_limits.csv'
product_resources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/product_resources.csv'
resource_limits_df = pd.read_csv(resource_limits_path, sep=',')
product_resources_df = pd.read_csv(product_resources_path, sep=',')
widget_names = [f'Widget{i}' for i in range(1, 142)]
product_resources_df['Product_norm'] = product_resources_df['Product'].apply(normalize_product)
product_resources_df.set_index('Product_norm', inplace=True)
missing_widgets = [w for w in widget_names if w not in product_resources_df.index]
if missing_widgets:
    raise ValueError(f'Missing widget(s) in product_resources.csv: {missing_widgets}')
LaborHours = {w: float(product_resources_df.loc[w, 'LaborHours']) for w in widget_names}
MaterialA = {w: float(product_resources_df.loc[w, 'MaterialA']) for w in widget_names}
MaterialB = {w: float(product_resources_df.loc[w, 'MaterialB']) for w in widget_names}
Profit = {w: float(product_resources_df.loc[w, 'Profit']) for w in widget_names}
resource_limits_df['Resource_norm'] = resource_limits_df['Resource'].apply(normalize_resource)
resource_limits_map = dict(zip(resource_limits_df['Resource_norm'], resource_limits_df['MonthlyLimit']))
resource_keys = {'LaborHours': normalize_resource('LaborHours'), 'MaterialA': normalize_resource('MaterialA'), 'MaterialB': normalize_resource('MaterialB')}
if not all((k in resource_limits_map for k in resource_keys.values())):
    raise ValueError('Missing required resource limits in resource_limits.csv')
LaborHours_limit = float(resource_limits_map[resource_keys['LaborHours']])
MaterialA_limit = float(resource_limits_map[resource_keys['MaterialA']])
MaterialB_limit = float(resource_limits_map[resource_keys['MaterialB']])
CATALYSTX_WIDGET = 'Widget3'
CATALYSTX_RATE = 5.0
CATALYSTX_SALE_PRICE = 300.0
CATALYSTX_DISPOSAL_COST = 200.0
CATALYSTX_SALES_CAP = 1500.0
m = gp.Model('AerospaceWidgetProduction')
x = m.addVars(widget_names, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
s = m.addVar(lb=0.0, ub=CATALYSTX_SALES_CAP, vtype=gp.GRB.CONTINUOUS, name='s')
d = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='d')
obj = gp.quicksum((Profit[w] * x[w] for w in widget_names)) + CATALYSTX_SALE_PRICE * s - CATALYSTX_DISPOSAL_COST * d
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((LaborHours[w] * x[w] for w in widget_names)) <= LaborHours_limit, name='labor')
m.addConstr(gp.quicksum((MaterialA[w] * x[w] for w in widget_names)) <= MaterialA_limit, name='matA')
m.addConstr(gp.quicksum((MaterialB[w] * x[w] for w in widget_names)) <= MaterialB_limit, name='matB')
m.addConstr(CATALYSTX_RATE * x[CATALYSTX_WIDGET] == s + d, name='catalystx_balance')
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal:.6f}')
    for w in widget_names:
        print(f'{x[w].VarName} {x[w].X:.6f}')
    print(f'{s.VarName} {s.X:.6f}')
    print(f'{d.VarName} {d.X:.6f}')
else:
    print(f'Solver status: {m.status}')