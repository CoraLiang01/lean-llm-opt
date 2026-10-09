import gurobipy as gp
import pandas as pd
import numpy as np
import re

def normalize_colnames(df):
    df.columns = [col.strip() for col in df.columns]
    return df

def solve_problem():
    resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/resource_limits.csv'
    product_resources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/product_resources.csv'
    df_limits = pd.read_csv(resource_limits_path, sep=',')
    df_limits = normalize_colnames(df_limits)
    df_products = pd.read_csv(product_resources_path, sep=',')
    df_products = normalize_colnames(df_products)
    widgets = [f'Widget{i}' for i in range(1, 142)]
    widgets_set = set(widgets)
    df_products['Product'] = df_products['Product'].astype(str).str.strip()
    products_in_file = set(df_products['Product'])
    missing_widgets = widgets_set - products_in_file
    if missing_widgets:
        raise ValueError(f'Missing widget(s) in product_resources.csv: {sorted(missing_widgets)}')
    labor_hours = {}
    material_a = {}
    material_b = {}
    profit = {}
    for (_, row) in df_products.iterrows():
        prod = row['Product'].strip()
        if prod in widgets_set:
            labor_hours[prod] = float(row['LaborHours'])
            material_a[prod] = float(row['MaterialA'])
            material_b[prod] = float(row['MaterialB'])
            profit[prod] = float(row['Profit'])
    for prod in widgets:
        if prod not in labor_hours or prod not in material_a or prod not in material_b or (prod not in profit):
            raise ValueError(f'Missing coefficients for {prod}')
    df_limits['Resource'] = df_limits['Resource'].astype(str).str.strip().str.casefold()
    resource_map = {r: int(lim) for (r, lim) in zip(df_limits['Resource'], df_limits['MonthlyLimit'])}

    def get_limit(key):
        for k in resource_map:
            if k.replace(' ', '') == key.casefold().replace(' ', ''):
                return resource_map[k]
        raise ValueError(f"Resource limit for '{key}' not found in resource_limits.csv")
    labor_limit = get_limit('LaborHours')
    material_a_limit = get_limit('MaterialA')
    material_b_limit = get_limit('MaterialB')
    m = gp.Model('WidgetProduction')
    x = m.addVars(widgets, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    s = m.addVar(lb=0.0, ub=1500.0, vtype=gp.GRB.CONTINUOUS, name='s')
    d = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='d')
    obj = gp.quicksum((profit[i] * x[i] for i in widgets)) + 300 * s - 200 * d
    m.setObjective(obj, gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((labor_hours[i] * x[i] for i in widgets)) <= labor_limit, name='labor')
    m.addConstr(gp.quicksum((material_a[i] * x[i] for i in widgets)) <= material_a_limit, name='matA')
    m.addConstr(gp.quicksum((material_b[i] * x[i] for i in widgets)) <= material_b_limit, name='matB')
    m.addConstr(5.0 * x['Widget3'] == s + d, name='catalyst_balance')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal:.6f}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X:.6f}')
else:
    print(f'Solver status: {m.status}')