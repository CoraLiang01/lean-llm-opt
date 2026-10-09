import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
products = ['Product I', 'Product II', 'Product III']
procedures = ['A', 'B']
equipments = ['A1', 'A2', 'B1', 'B2', 'B3']
equipment_procedure = {'A1': 'A', 'A2': 'A', 'B1': 'B', 'B2': 'B', 'B3': 'B'}
eligible_equipment = {('Product I', 'A'): ['A1', 'A2'], ('Product I', 'B'): ['B1', 'B2', 'B3'], ('Product II', 'A'): ['A1', 'A2'], ('Product II', 'B'): ['B1'], ('Product III', 'A'): ['A2'], ('Product III', 'B'): ['B2']}

def row_idx(label):
    idx = df.index[df['Equipment / Cost'].str.strip() == label]
    if len(idx) == 0:
        raise KeyError(f"Row '{label}' not found in CSV.")
    return idx[0]
unit_price_row = row_idx('Unit Price (yuan/unit)')
selling_price = {}
for p in products:
    val = df.at[unit_price_row, p].strip()
    selling_price[p] = float(val)
raw_material_row = row_idx('Raw Material Cost (yuan/unit)')
raw_material_cost = {}
for p in products:
    val = df.at[raw_material_row, p].strip()
    raw_material_cost[p] = float(val)
processing_time = {}
available_time = {}
equipment_full_cost = {}
for e in equipments:
    e_row = row_idx(e)
    avail_time_str = df.at[e_row, 'Available Equipment Operating Time'].strip()
    if avail_time_str == '':
        raise ValueError(f'Missing available time for equipment {e}')
    available_time[e] = float(avail_time_str)
    equip_cost_str = df.at[e_row, 'Equipment Cost at Full Load (yuan)'].strip()
    if equip_cost_str == '':
        raise ValueError(f'Missing equipment cost at full load for {e}')
    equipment_full_cost[e] = float(equip_cost_str)
    for p in products:
        pt_str = df.at[e_row, p].strip()
        if pt_str == '':
            continue
        processing_time[p, e] = float(pt_str)
m = gp.Model('ProductionPlan')
x_vars = {}
for p in products:
    for proc in procedures:
        for e in eligible_equipment[p, proc]:
            if (p, e) in processing_time:
                x_vars[p, e] = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name=f"x_{p.replace(' ', '')}_{e}")
for p in products:
    sum_A = gp.quicksum((x_vars[p, e] for e in eligible_equipment[p, 'A'] if (p, e) in x_vars))
    sum_B = gp.quicksum((x_vars[p, e] for e in eligible_equipment[p, 'B'] if (p, e) in x_vars))
    m.addConstr(sum_A == sum_B, name=f"flow_{p.replace(' ', '')}")
for e in equipments:
    used_time = gp.quicksum((processing_time[p, e] * x_vars[p, e] for p in products if (p, e) in x_vars))
    m.addConstr(used_time <= available_time[e], name=f'cap_{e}')
profit_expr = gp.LinExpr()
for p in products:
    unit_profit = selling_price[p] - raw_material_cost[p]
    total_prod = gp.quicksum((x_vars[p, e] for e in eligible_equipment[p, 'A'] if (p, e) in x_vars))
    profit_expr += unit_profit * total_prod
equip_cost_expr = gp.LinExpr()
for e in equipments:
    used_time = gp.quicksum((processing_time[p, e] * x_vars[p, e] for p in products if (p, e) in x_vars))
    equip_cost_expr += equipment_full_cost[e] * (used_time / available_time[e])
m.setObjective(profit_expr - equip_cost_expr, gp.GRB.MAXIMIZE)
m.optimize()