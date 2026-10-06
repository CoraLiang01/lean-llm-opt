import gurobipy as gp
import pandas as pd
import numpy as np
import re
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/resource_limits.csv'
product_resources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/product_resources.csv'
resource_limits = pd.read_csv(resource_limits_path, sep=',')
product_resources = pd.read_csv(product_resources_path, sep=',')
widget_ids = [f'Widget{i}' for i in range(1, 142)]
product_resources['Product'] = product_resources['Product'].astype(str)
missing_widgets = [wid for wid in widget_ids if wid not in set(product_resources['Product'])]
if missing_widgets:
    raise ValueError(f'Missing widget(s) in product_resources.csv: {missing_widgets}')
labor_hours = {}
material_a = {}
material_b = {}
profit = {}
for wid in widget_ids:
    row = product_resources.loc[product_resources['Product'] == wid]
    if row.shape[0] != 1:
        raise ValueError(f'Widget {wid} not found exactly once in product_resources.csv')
    labor_hours[wid] = float(row['LaborHours'].values[0])
    material_a[wid] = float(row['MaterialA'].values[0])
    material_b[wid] = float(row['MaterialB'].values[0])
    profit[wid] = float(row['Profit'].values[0])
resource_limits['Resource'] = resource_limits['Resource'].astype(str)

def get_limit(resource_name):
    matches = resource_limits[resource_limits['Resource'].str.casefold() == resource_name.casefold()]
    if matches.shape[0] != 1:
        raise ValueError(f"Resource '{resource_name}' not found exactly once in resource_limits.csv")
    return float(matches['MonthlyLimit'].values[0])
labor_limit = get_limit('LaborHours')
material_a_limit = get_limit('MaterialA')
material_b_limit = get_limit('MaterialB')

def solve_problem():
    m = gp.Model('WidgetProduction')
    m.Params.MIPGap = 0.0001
    x = m.addVars(widget_ids, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    s = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='s')
    d = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='d')
    obj = gp.quicksum((profit[wid] * x[wid] for wid in widget_ids)) + 300 * s - 200 * d
    m.setObjective(obj, gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((labor_hours[wid] * x[wid] for wid in widget_ids)) <= labor_limit, name='labor')
    m.addConstr(gp.quicksum((material_a[wid] * x[wid] for wid in widget_ids)) <= material_a_limit, name='matA')
    m.addConstr(gp.quicksum((material_b[wid] * x[wid] for wid in widget_ids)) <= material_b_limit, name='matB')
    m.addConstr(5.0 * x['Widget3'] == s + d, name='catalyst_balance')
    m.addConstr(s <= 1500.0, name='catalyst_sales_cap')
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')