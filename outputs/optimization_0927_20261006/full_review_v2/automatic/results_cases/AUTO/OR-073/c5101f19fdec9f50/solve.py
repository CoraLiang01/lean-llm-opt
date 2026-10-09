import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv'
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
products = ['Product I', 'Product II', 'Product III']
product_short = {'Product I': 'I', 'Product II': 'II', 'Product III': 'III'}
procedures = ['A', 'B']
equipment_rows = df[~df['Equipment / Cost'].str.strip().str.casefold().isin(['raw material cost (yuan/unit)', 'unit price (yuan/unit)']) & (df['Equipment / Cost'].str.strip() != '')].copy()
equipment_names = equipment_rows['Equipment / Cost'].str.strip()
A_equipment = [ename for ename in equipment_names if ename.upper().startswith('A')]
B_equipment = [ename for ename in equipment_names if ename.upper().startswith('B')]
equipment_procedure = {}
for ename in equipment_names:
    if ename.upper().startswith('A'):
        equipment_procedure[ename] = 'A'
    elif ename.upper().startswith('B'):
        equipment_procedure[ename] = 'B'
eligibility = {}
eligibility['Product I', 'A'] = [e for e in A_equipment if e in ['A1', 'A2']]
eligibility['Product II', 'A'] = list(A_equipment)
eligibility['Product III', 'A'] = [e for e in A_equipment if e == 'A2']
eligibility['Product I', 'B'] = list(B_equipment)
eligibility['Product II', 'B'] = [e for e in B_equipment if e == 'B1']
eligibility['Product III', 'B'] = [e for e in B_equipment if e == 'B2']

def parse_float(val):
    try:
        return float(val)
    except Exception:
        return np.nan
proc_time = {}
for (idx, row) in equipment_rows.iterrows():
    eq = row['Equipment / Cost'].strip()
    for prod in products:
        val = parse_float(row[prod])
        if not np.isnan(val):
            proc_time[eq, prod] = val
avail_time = {}
for (idx, row) in equipment_rows.iterrows():
    eq = row['Equipment / Cost'].strip()
    val = parse_float(row['Available Equipment Operating Time'])
    if not np.isnan(val):
        avail_time[eq] = val
equip_cost_full = {}
for (idx, row) in equipment_rows.iterrows():
    eq = row['Equipment / Cost'].strip()
    val = parse_float(row['Equipment Cost at Full Load (yuan)'])
    if not np.isnan(val):
        equip_cost_full[eq] = val
raw_mat_row = df['Equipment / Cost'].str.strip().str.casefold() == 'raw material cost (yuan/unit)'
raw_mat_cost = {}
if raw_mat_row.any():
    row = df[raw_mat_row].iloc[0]
    for prod in products:
        val = parse_float(row[prod])
        if not np.isnan(val):
            raw_mat_cost[prod] = val
unit_price_row = df['Equipment / Cost'].str.strip().str.casefold() == 'unit price (yuan/unit)'
unit_price = {}
if unit_price_row.any():
    row = df[unit_price_row].iloc[0]
    for prod in products:
        val = parse_float(row[prod])
        if not np.isnan(val):
            unit_price[prod] = val
x_vars = {}
for prod in products:
    for proc in procedures:
        for eq in eligibility.get((prod, proc), []):
            x_vars[prod, proc, eq] = None
m = gp.Model('ProductionPlan')
for key in x_vars:
    (prod, proc, eq) = key
    x_vars[key] = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name=f'x_{product_short[prod]}_{proc}_{eq}')
m.update()
for prod in products:
    sum_A = gp.quicksum((x_vars[prod, 'A', eq] for eq in eligibility.get((prod, 'A'), []) if (prod, 'A', eq) in x_vars))
    sum_B = gp.quicksum((x_vars[prod, 'B', eq] for eq in eligibility.get((prod, 'B'), []) if (prod, 'B', eq) in x_vars))
    m.addConstr(sum_A == sum_B, name=f'flow_{product_short[prod]}')
for eq in equipment_names:
    relevant_keys = [key for key in x_vars if key[2] == eq]
    if eq in avail_time:
        m.addConstr(gp.quicksum((proc_time.get((eq, key[0]), 0.0) * x_vars[key] for key in relevant_keys)) <= avail_time[eq], name=f'equip_time_{eq}')
prod_total = {}
for prod in products:
    prod_total[prod] = gp.quicksum((x_vars[prod, 'A', eq] for eq in eligibility.get((prod, 'A'), []) if (prod, 'A', eq) in x_vars))
total_revenue = gp.quicksum((unit_price[prod] * prod_total[prod] for prod in products))
total_raw_mat_cost = gp.quicksum((raw_mat_cost[prod] * prod_total[prod] for prod in products))
equip_operating_cost = []
for eq in equipment_names:
    if eq in equip_cost_full and eq in avail_time and (avail_time[eq] > 0):
        relevant_keys = [key for key in x_vars if key[2] == eq]
        usage_time = gp.quicksum((proc_time.get((eq, key[0]), 0.0) * x_vars[key] for key in relevant_keys))
        equip_operating_cost.append(equip_cost_full[eq] * (usage_time / avail_time[eq]))
total_equip_cost = gp.quicksum(equip_operating_cost)
m.setObjective(total_revenue - total_raw_mat_cost - total_equip_cost, gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total profit: {m.objVal:.2f}\n')
    print('Production Plan (units per product):')
    for prod in products:
        qty = prod_total[prod].getValue()
        print(f'  {prod}: {qty:.2f}')
    print('\nDetailed assignment (units processed on each equipment):')
    for (key, var) in x_vars.items():
        if var.X > 1e-06:
            (prod, proc, eq) = key
            print(f'  {prod}, Procedure {proc}, Equipment {eq}: {var.X:.2f}')
    print('\nEquipment usage (hours):')
    for eq in equipment_names:
        if eq in avail_time:
            relevant_keys = [key for key in x_vars if key[2] == eq]
            usage = sum((proc_time.get((eq, key[0]), 0.0) * x_vars[key].X for key in relevant_keys))
            print(f'  {eq}: {usage:.2f} / {avail_time[eq]:.2f}')
    print('\nEquipment operating costs (yuan):')
    for eq in equipment_names:
        if eq in equip_cost_full and eq in avail_time and (avail_time[eq] > 0):
            relevant_keys = [key for key in x_vars if key[2] == eq]
            usage = sum((proc_time.get((eq, key[0]), 0.0) * x_vars[key].X for key in relevant_keys))
            cost = equip_cost_full[eq] * (usage / avail_time[eq])
            print(f'  {eq}: {cost:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')