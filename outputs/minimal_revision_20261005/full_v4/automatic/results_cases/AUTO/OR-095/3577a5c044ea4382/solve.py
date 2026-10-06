import gurobipy as gp
import pandas as pd
import numpy as np
import re

def normalize_resource(s):
    return re.sub('\\s+', '', str(s)).casefold()

def normalize_product(s):
    return str(s).strip()

def solve_problem():
    resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/resource_limits.csv'
    df_limits = pd.read_csv(resource_limits_path, sep=',')
    df_limits['Resource_norm'] = df_limits['Resource'].apply(normalize_resource)
    resource_map = dict(zip(df_limits['Resource_norm'], df_limits['MonthlyLimit']))
    product_resources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/product_resources.csv'
    df_prod = pd.read_csv(product_resources_path, sep=',')
    widget_names = [f'Widget{i}' for i in range(1, 142)]
    df_prod['Product_norm'] = df_prod['Product'].apply(normalize_product)
    prod_idx = set(df_prod['Product_norm'])
    missing_widgets = [w for w in widget_names if w not in prod_idx]
    if missing_widgets:
        raise ValueError(f'Missing widget(s) in product_resources.csv: {missing_widgets}')
    labor_hours = {}
    material_a = {}
    material_b = {}
    profit = {}
    for w in widget_names:
        row = df_prod.loc[df_prod['Product_norm'] == w]
        if row.shape[0] != 1:
            raise ValueError(f'Widget {w} not found uniquely in product_resources.csv')
        labor_hours[w] = float(row['LaborHours'].values[0])
        material_a[w] = float(row['MaterialA'].values[0])
        material_b[w] = float(row['MaterialB'].values[0])
        profit[w] = float(row['Profit'].values[0])
    labor_limit = float(resource_map[normalize_resource('LaborHours')])
    material_a_limit = float(resource_map[normalize_resource('MaterialA')])
    material_b_limit = float(resource_map[normalize_resource('MaterialB')])
    m = gp.Model('WidgetProduction')
    x = m.addVars(widget_names, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    s = m.addVar(lb=0.0, ub=1500.0, vtype=gp.GRB.CONTINUOUS, name='s')
    d = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='d')
    obj = gp.quicksum((profit[w] * x[w] for w in widget_names)) + 300.0 * s - 200.0 * d
    m.setObjective(obj, gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((labor_hours[w] * x[w] for w in widget_names)) <= labor_limit, name='labor')
    m.addConstr(gp.quicksum((material_a[w] * x[w] for w in widget_names)) <= material_a_limit, name='matA')
    m.addConstr(gp.quicksum((material_b[w] * x[w] for w in widget_names)) <= material_b_limit, name='matB')
    m.addConstr(s + d == 5.0 * x['Widget3'], name='catalyst_balance')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')