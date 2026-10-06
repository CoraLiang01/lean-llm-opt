import gurobipy as gp
import pandas as pd
import numpy as np
import re
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/resource_limits.csv'
product_resources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/product_resources.csv'

def solve_problem():
    resource_limits = pd.read_csv(resource_limits_path, sep=',')
    resource_limits['Resource_norm'] = resource_limits['Resource'].astype(str).str.casefold()
    resource_limits = resource_limits.set_index('Resource_norm')
    product_resources = pd.read_csv(product_resources_path, sep=',')
    product_resources['Product_norm'] = product_resources['Product'].astype(str).str.casefold()
    product_resources = product_resources.set_index('Product_norm')
    widget_names = [f'Widget{i}' for i in range(1, 142)]
    widget_names_norm = [w.casefold() for w in widget_names]
    missing_widgets = [w for w in widget_names_norm if w not in product_resources.index]
    if missing_widgets:
        raise ValueError(f'Missing widget(s) in product_resources.csv: {missing_widgets}')
    labor_hours = {w: float(product_resources.loc[w, 'LaborHours']) for w in widget_names_norm}
    material_a = {w: float(product_resources.loc[w, 'MaterialA']) for w in widget_names_norm}
    material_b = {w: float(product_resources.loc[w, 'MaterialB']) for w in widget_names_norm}
    profit = {w: float(product_resources.loc[w, 'Profit']) for w in widget_names_norm}
    labor_limit = float(resource_limits.loc['labor hours', 'MonthlyLimit'])
    material_a_limit = float(resource_limits.loc['material a', 'MonthlyLimit'])
    material_b_limit = float(resource_limits.loc['material b', 'MonthlyLimit'])
    widget3_norm = 'widget3'
    if widget3_norm not in widget_names_norm:
        raise ValueError('Widget3 not found in widget_names_norm.')
    m = gp.Model('AerospaceWidgetProduction')
    x = m.addVars(widget_names_norm, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    s = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='s')
    d = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='d')
    obj = gp.quicksum((profit[w] * x[w] for w in widget_names_norm)) + 300 * s - 200 * d
    m.setObjective(obj, gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((labor_hours[w] * x[w] for w in widget_names_norm)) <= labor_limit, name='labor')
    m.addConstr(gp.quicksum((material_a[w] * x[w] for w in widget_names_norm)) <= material_a_limit, name='matA')
    m.addConstr(gp.quicksum((material_b[w] * x[w] for w in widget_names_norm)) <= material_b_limit, name='matB')
    m.addConstr(5 * x[widget3_norm] == s + d, name='catalyst_balance')
    m.addConstr(s <= 1500, name='catalyst_sales_cap')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        print(f'ObjVal {m.objVal}')
        for w in widget_names_norm:
            print(f'{x[w].VarName} {x[w].X}')
        print(f'{s.VarName} {s.X}')
        print(f'{d.VarName} {d.X}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_problem()