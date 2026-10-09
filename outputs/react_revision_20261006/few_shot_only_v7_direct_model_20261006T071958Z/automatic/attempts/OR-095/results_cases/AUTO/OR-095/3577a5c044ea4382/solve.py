import gurobipy as gp
import pandas as pd
import numpy as np
import re
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/resource_limits.csv'
product_resources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/product_resources.csv'

def solve_problem():
    resource_limits_df = pd.read_csv(resource_limits_path, sep=',', dtype=str, keep_default_na=False)
    product_resources_df = pd.read_csv(product_resources_path, sep=',', dtype=str, keep_default_na=False)
    widget_ids = [f'Widget{i}' for i in range(1, 142)]
    product_resources_df['Product_norm'] = product_resources_df['Product'].astype(str).str.strip()
    available_widgets = set(product_resources_df['Product_norm'])
    missing_widgets = [wid for wid in widget_ids if wid not in available_widgets]
    if missing_widgets:
        raise ValueError(f'Missing widget(s) in product_resources.csv: {missing_widgets}')
    for col in ['LaborHours', 'MaterialA', 'MaterialB', 'Profit']:
        product_resources_df[col] = pd.to_numeric(product_resources_df[col], errors='raise')
    labor_hours = dict(zip(product_resources_df['Product_norm'], product_resources_df['LaborHours']))
    material_a = dict(zip(product_resources_df['Product_norm'], product_resources_df['MaterialA']))
    material_b = dict(zip(product_resources_df['Product_norm'], product_resources_df['MaterialB']))
    profit = dict(zip(product_resources_df['Product_norm'], product_resources_df['Profit']))
    resource_limits_df['Resource_norm'] = resource_limits_df['Resource'].astype(str).str.strip().str.casefold()
    resource_limits_df['MonthlyLimit'] = pd.to_numeric(resource_limits_df['MonthlyLimit'], errors='raise')
    resource_map = {}
    for (_, row) in resource_limits_df.iterrows():
        resource_map[row['Resource_norm']] = row['MonthlyLimit']
    labor_resource = None
    materiala_resource = None
    materialb_resource = None
    for res in resource_map:
        if re.fullmatch('labor\\s*hours?', res, re.IGNORECASE):
            labor_resource = res
        elif re.fullmatch('material\\s*a', res, re.IGNORECASE):
            materiala_resource = res
        elif re.fullmatch('material\\s*b', res, re.IGNORECASE):
            materialb_resource = res
    if labor_resource is None or materiala_resource is None or materialb_resource is None:
        raise ValueError('Could not find all required resources in resource_limits.csv')
    labor_limit = resource_map[labor_resource]
    materiala_limit = resource_map[materiala_resource]
    materialb_limit = resource_map[materialb_resource]
    m = gp.Model('WidgetProduction')
    quantity_vars = m.addVars(widget_ids, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    s_var = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='s')
    d_var = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='d')
    obj_expr = gp.quicksum((profit[i] * quantity_vars[i] for i in widget_ids)) + 300 * s_var - 200 * d_var
    m.setObjective(obj_expr, gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((labor_hours[i] * quantity_vars[i] for i in widget_ids)) <= labor_limit, name='labor')
    m.addConstr(gp.quicksum((material_a[i] * quantity_vars[i] for i in widget_ids)) <= materiala_limit, name='materialA')
    m.addConstr(gp.quicksum((material_b[i] * quantity_vars[i] for i in widget_ids)) <= materialb_limit, name='materialB')
    m.addConstr(5.0 * quantity_vars['Widget3'] == s_var + d_var, name='catalyst_balance')
    m.addConstr(s_var <= 1500.0, name='catalyst_sales_cap')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')