import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
products = ['Product I', 'Product II', 'Product III']
procedures = {'A': ['A1', 'A2'], 'B': ['B1', 'B2', 'B3']}
equipments = ['A1', 'A2', 'B1', 'B2', 'B3']
equipment_procedure = {}
for e in equipments:
    if e.startswith('A'):
        equipment_procedure[e] = 'A'
    elif e.startswith('B'):
        equipment_procedure[e] = 'B'
    else:
        raise ValueError(f'Unknown equipment type: {e}')

def get_row_value(row_name, col_name):
    row = df[df['Equipment / Cost'] == row_name]
    if row.empty:
        raise KeyError(f"Row '{row_name}' not found in CSV.")
    val = row.iloc[0][col_name]
    if val == '':
        raise ValueError(f'Missing value for {row_name}, {col_name}')
    return float(val)
selling_price = {}
raw_material_cost = {}
for p in products:
    selling_price[p] = get_row_value('Unit Price (yuan/unit)', p)
    raw_material_cost[p] = get_row_value('Raw Material Cost (yuan/unit)', p)
processing_time = {}
eligible_pairs = set()
for e in equipments:
    proc = equipment_procedure[e]
    row = df[df['Equipment / Cost'] == e]
    if row.empty:
        raise KeyError(f"Equipment row '{e}' not found in CSV.")
    for p in products:
        val = row.iloc[0][p]
        if val.strip() == '':
            continue
        allowed = False
        if p == 'Product I':
            allowed = True
        elif p == 'Product II':
            if proc == 'A':
                allowed = True
            elif proc == 'B' and e == 'B1':
                allowed = True
        elif p == 'Product III':
            if e == 'A2' or e == 'B2':
                allowed = True
        if allowed:
            processing_time[p, e] = float(val)
            eligible_pairs.add((p, e))
equipment_operating_time = {}
equipment_full_load_cost = {}
for e in equipments:
    row = df[df['Equipment / Cost'] == e]
    if row.empty:
        raise KeyError(f"Equipment row '{e}' not found in CSV.")
    avail_time = row.iloc[0]['Available Equipment Operating Time']
    if avail_time.strip() == '':
        raise ValueError(f'Missing available operating time for equipment {e}')
    equipment_operating_time[e] = float(avail_time)
    full_load_cost = row.iloc[0]['Equipment Cost at Full Load (yuan)']
    if full_load_cost.strip() == '':
        raise ValueError(f'Missing full load cost for equipment {e}')
    equipment_full_load_cost[e] = float(full_load_cost)
m = gp.Model('ProductionPlan')
x_vars = m.addVars(eligible_pairs, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
equipment_usage_expr = {}
for e in equipments:
    usage = gp.quicksum((processing_time[p, e] * x_vars[p, e] for p in products if (p, e) in x_vars))
    equipment_usage_expr[e] = usage
total_revenue = gp.quicksum((selling_price[p] * x_vars[p, e] for (p, e) in x_vars))
total_raw_material_cost = gp.quicksum((raw_material_cost[p] * x_vars[p, e] for (p, e) in x_vars))
total_equipment_cost = gp.quicksum((equipment_full_load_cost[e] * (equipment_usage_expr[e] / equipment_operating_time[e]) for e in equipments))
m.setObjective(total_revenue - total_raw_material_cost - total_equipment_cost, gp.GRB.MAXIMIZE)
for p in products:
    A_equip = [e for e in procedures['A'] if (p, e) in x_vars]
    B_equip = [e for e in procedures['B'] if (p, e) in x_vars]
    lhs = gp.quicksum((x_vars[p, e] for e in A_equip))
    rhs = gp.quicksum((x_vars[p, e] for e in B_equip))
    m.addConstr(lhs == rhs, name=f'sync_{p}')
for e in equipments:
    usage = equipment_usage_expr[e]
    m.addConstr(usage <= equipment_operating_time[e], name=f'cap_{e}')
m.optimize()