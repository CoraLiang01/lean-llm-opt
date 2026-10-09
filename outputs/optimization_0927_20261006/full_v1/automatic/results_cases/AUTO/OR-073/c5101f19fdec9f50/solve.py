import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv'
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
products = ['I', 'II', 'III']
product_col_map = {'I': 'Product I', 'II': 'Product II', 'III': 'Product III'}
procedures = ['A', 'B']
equip_A = ['A1', 'A2']
equip_B = ['B1', 'B2', 'B3']
equip_all = equip_A + equip_B
equipment_rows = df['Equipment / Cost'].isin(equip_all)
equip_df = df.loc[equipment_rows, :].copy()
equip_df['Equipment'] = equip_df['Equipment / Cost'].str.strip()
for col in ['Product I', 'Product II', 'Product III', 'Available Equipment Operating Time', 'Equipment Cost at Full Load (yuan)']:
    equip_df[col] = pd.to_numeric(equip_df[col], errors='coerce')
proc_time = {}
for e in equip_all:
    row = equip_df.loc[equip_df['Equipment'] == e]
    if row.empty:
        raise ValueError(f'Missing equipment row for {e}')
    for p in products:
        col = product_col_map[p]
        val = float(row.iloc[0][col]) if pd.notnull(row.iloc[0][col]) and row.iloc[0][col] != '' else None
        proc_time[e, p] = val
equip_avail_time = {}
equip_full_cost = {}
for e in equip_all:
    row = equip_df.loc[equip_df['Equipment'] == e]
    if row.empty:
        raise ValueError(f'Missing equipment row for {e}')
    avail_time = row.iloc[0]['Available Equipment Operating Time']
    full_cost = row.iloc[0]['Equipment Cost at Full Load (yuan)']
    equip_avail_time[e] = float(avail_time) if pd.notnull(avail_time) and avail_time != '' else None
    equip_full_cost[e] = float(full_cost) if pd.notnull(full_cost) and full_cost != '' else None

def get_row_value(row_name, col):
    row = df.loc[df['Equipment / Cost'].str.strip() == row_name]
    if row.empty:
        raise ValueError(f"Missing row '{row_name}' for column '{col}'")
    val = row.iloc[0][col]
    return float(val) if val != '' else None
raw_material_cost = {}
unit_price = {}
for p in products:
    col = product_col_map[p]
    raw_material_cost[p] = get_row_value('Raw Material Cost (yuan/unit)', col)
    unit_price[p] = get_row_value('Unit Price (yuan/unit)', col)
eligible = {}
for p in products:
    for proc in procedures:
        if proc == 'A':
            for e in equip_A:
                if p == 'I' or p == 'II':
                    eligible[p, proc, e] = True
                elif p == 'III' and e == 'A2':
                    eligible[p, proc, e] = True
                else:
                    eligible[p, proc, e] = False
        elif proc == 'B':
            for e in equip_B:
                if p == 'I':
                    eligible[p, proc, e] = True
                elif p == 'II' and e == 'B1':
                    eligible[p, proc, e] = True
                elif p == 'III' and e == 'B2':
                    eligible[p, proc, e] = True
                else:
                    eligible[p, proc, e] = False
m = gp.Model('FactoryProductionPlan')
x_vars = {}
for p in products:
    for proc in procedures:
        if proc == 'A':
            eqs = equip_A
        else:
            eqs = equip_B
        for e in eqs:
            if eligible[p, proc, e] and proc_time[e, p] is not None:
                x_vars[p, proc, e] = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name=f'x_{p}_{proc}_{e}')
for p in products:
    sum_A = gp.quicksum((x_vars[p, 'A', e] for e in equip_A if (p, 'A', e) in x_vars))
    sum_B = gp.quicksum((x_vars[p, 'B', e] for e in equip_B if (p, 'B', e) in x_vars))
    m.addConstr(sum_A == sum_B, name=f'proc_balance_{p}')
for e in equip_all:
    usage = []
    for p in products:
        for proc in procedures:
            if (p, proc, e) in eligible and eligible[p, proc, e] and ((e, p) in proc_time) and (proc_time[e, p] is not None):
                key = (p, proc, e)
                if key in x_vars:
                    usage.append(proc_time[e, p] * x_vars[key])
    if equip_avail_time[e] is not None:
        m.addConstr(gp.quicksum(usage) <= equip_avail_time[e], name=f'time_limit_{e}')
prod_total = {}
for p in products:
    prod_total[p] = gp.quicksum((x_vars[p, 'A', e] for e in equip_A if (p, 'A', e) in x_vars))
total_revenue = gp.quicksum((unit_price[p] * prod_total[p] for p in products))
total_raw_cost = gp.quicksum((raw_material_cost[p] * prod_total[p] for p in products))
equip_cost_terms = []
for e in equip_all:
    usage = []
    for p in products:
        for proc in procedures:
            if (p, proc, e) in eligible and eligible[p, proc, e] and ((e, p) in proc_time) and (proc_time[e, p] is not None):
                key = (p, proc, e)
                if key in x_vars:
                    usage.append(proc_time[e, p] * x_vars[key])
    if equip_avail_time[e] is not None and equip_full_cost[e] is not None:
        equip_cost_terms.append(equip_full_cost[e] * (gp.quicksum(usage) / equip_avail_time[e]))
total_equip_cost = gp.quicksum(equip_cost_terms)
m.setObjective(total_revenue - total_raw_cost - total_equip_cost, gp.GRB.MAXIMIZE)
m.optimize()