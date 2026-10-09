import gurobipy as gp
import pandas as pd
import numpy as np
import re
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/resource_limits.csv'
product_resources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/product_resources.csv'

def solve_problem():
    resource_limits_df = pd.read_csv(resource_limits_path, dtype=str, keep_default_na=False)
    product_resources_df = pd.read_csv(product_resources_path, dtype=str, keep_default_na=False)
    widget_ids = [f'Widget{i}' for i in range(1, 142)]
    product_resources_df['Product_norm'] = product_resources_df['Product'].astype(str).str.strip()
    missing_widgets = [wid for wid in widget_ids if wid not in set(product_resources_df['Product_norm'])]
    if missing_widgets:
        raise ValueError(f'Missing widget(s) in product_resources.csv: {missing_widgets}')
    labor_hours = {}
    material_a = {}
    material_b = {}
    profit = {}
    for (_, row) in product_resources_df.iterrows():
        wid = row['Product_norm']
        if wid in widget_ids:
            try:
                labor_hours[wid] = float(row['LaborHours'])
                material_a[wid] = float(row['MaterialA'])
                material_b[wid] = float(row['MaterialB'])
                profit[wid] = float(row['Profit'])
            except Exception as e:
                raise ValueError(f'Non-numeric parameter for widget {wid}: {e}')
    resource_limits_df['Resource_norm'] = resource_limits_df['Resource'].astype(str).str.strip().str.casefold()
    resource_limits = {}
    for (_, row) in resource_limits_df.iterrows():
        res = row['Resource_norm']
        try:
            resource_limits[res] = float(row['MonthlyLimit'])
        except Exception as e:
            raise ValueError(f"Non-numeric MonthlyLimit for resource {row['Resource']}: {e}")

    def find_resource_limit(name, fallback):
        for k in resource_limits:
            if name.casefold() in k:
                return resource_limits[k]
        return fallback
    labor_limit = find_resource_limit('labor', 5000.0)
    material_a_limit = find_resource_limit('material a', 24000.0)
    material_b_limit = find_resource_limit('material b', 15000.0)
    widget3_id = 'Widget3'
    if widget3_id not in widget_ids:
        raise ValueError('Widget3 not found in widget_ids.')
    m = gp.Model('AerospaceWidgetProduction')
    quantity_vars = m.addVars(widget_ids, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    s_var = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='s')
    d_var = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='d')
    obj_expr = gp.quicksum((profit[wid] * quantity_vars[wid] for wid in widget_ids)) + 300.0 * s_var - 200.0 * d_var
    m.setObjective(obj_expr, gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((labor_hours[wid] * quantity_vars[wid] for wid in widget_ids)) <= labor_limit, name='labor')
    m.addConstr(gp.quicksum((material_a[wid] * quantity_vars[wid] for wid in widget_ids)) <= material_a_limit, name='matA')
    m.addConstr(gp.quicksum((material_b[wid] * quantity_vars[wid] for wid in widget_ids)) <= material_b_limit, name='matB')
    m.addConstr(5.0 * quantity_vars[widget3_id] == s_var + d_var, name='catalyst_balance')
    m.addConstr(s_var <= 1500.0, name='catalyst_sales_cap')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')