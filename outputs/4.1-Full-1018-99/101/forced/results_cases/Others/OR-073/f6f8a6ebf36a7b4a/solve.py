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
equipment_ids = equipment_rows['Equipment / Cost'].tolist()
equipment_to_proc = {}
for eq in equipment_ids:
    eq_stripped = eq.strip()
    if re.match('^A\\d+$', eq_stripped, re.IGNORECASE):
        equipment_to_proc[eq_stripped] = 'A'
    elif re.match('^B\\d+$', eq_stripped, re.IGNORECASE):
        equipment_to_proc[eq_stripped] = 'B'
    else:
        continue
eligible_equipment = {}
for p in products:
    for proc in procedures:
        eligible_equipment[p, proc] = set()
eligible_equipment['Product I', 'A'] = set(['A1', 'A2'])
eligible_equipment['Product I', 'B'] = set(['B1', 'B2', 'B3'])
eligible_equipment['Product II', 'A'] = set(['A1', 'A2'])
eligible_equipment['Product II', 'B'] = set(['B1'])
eligible_equipment['Product III', 'A'] = set(['A2'])
eligible_equipment['Product III', 'B'] = set(['B2'])
processing_time = {}
for idx, row in equipment_rows.iterrows():
    eq = row['Equipment / Cost'].strip()
    for p in products:
        for proc in procedures:
            if eq in eligible_equipment[p, proc]:
                val = row[p].strip()
                if val == '':
                    raise ValueError(f'Missing processing time for {p} on {eq}')
                processing_time[p, eq] = float(val)
equipment_operating_time = {}
equipment_full_load_cost = {}
for idx, row in equipment_rows.iterrows():
    eq = row['Equipment / Cost'].strip()
    if eq not in equipment_to_proc:
        continue
    avail_time = row['Available Equipment Operating Time'].strip()
    cost_full_load = row['Equipment Cost at Full Load (yuan)'].strip()
    if avail_time == '' or cost_full_load == '':
        raise ValueError(f'Missing available time or cost for equipment {eq}')
    equipment_operating_time[eq] = float(avail_time)
    equipment_full_load_cost[eq] = float(cost_full_load)

def find_row_by_label(label):
    mask = df['Equipment / Cost'].str.strip().str.casefold() == label.strip().casefold()
    matches = df[mask]
    if matches.empty:
        raise ValueError(f"Row with label '{label}' not found in 'Equipment / Cost'")
    return matches.iloc[0]
raw_material_cost_row = find_row_by_label('Raw Material Cost (yuan/unit)')
unit_price_row = find_row_by_label('Unit Price (yuan/unit)')
raw_material_cost = {}
unit_price = {}
for p in products:
    val_rm = raw_material_cost_row[p].strip()
    val_up = unit_price_row[p].strip()
    if val_rm == '' or val_up == '':
        raise ValueError(f'Missing raw material cost or unit price for {p}')
    raw_material_cost[p] = float(val_rm)
    unit_price[p] = float(val_up)
m = gp.Model('FactoryProductionPlan')
x_vars = {}
for p in products:
    for proc in procedures:
        for e in eligible_equipment[p, proc]:
            x_vars[p, e] = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name=f'x_{product_short[p]}_{e}')
for p in products:
    sum_A = gp.quicksum((x_vars[p, e] for e in eligible_equipment[p, 'A']))
    sum_B = gp.quicksum((x_vars[p, e] for e in eligible_equipment[p, 'B']))
    m.addConstr(sum_A == sum_B, name=f'proc_balance_{product_short[p]}')
for e in equipment_operating_time:
    total_time = gp.quicksum((processing_time[p, e] * x_vars[p, e] for p in products for proc in procedures if e in eligible_equipment[p, proc] and (p, e) in x_vars))
    m.addConstr(total_time <= equipment_operating_time[e], name=f'equip_cap_{e}')
total_produced = {}
for p in products:
    total_produced[p] = gp.quicksum((x_vars[p, e] for e in eligible_equipment[p, 'A']))
total_revenue = gp.quicksum((unit_price[p] * total_produced[p] for p in products))
total_rm_cost = gp.quicksum((raw_material_cost[p] * total_produced[p] for p in products))
total_equip_cost = gp.quicksum((equipment_full_load_cost[e] * (gp.quicksum((processing_time[p, e] * x_vars[p, e] for p in products for proc in procedures if e in eligible_equipment[p, proc] and (p, e) in x_vars)) / equipment_operating_time[e]) for e in equipment_operating_time))
m.setObjective(total_revenue - total_rm_cost - total_equip_cost, gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total profit: {m.objVal:.2f}')
    print('\n--- Production Plan ---')
    for p in products:
        prod_qty = total_produced[p].getValue()
        print(f'Product {product_short[p]}: {prod_qty:.2f} units')
        for proc in procedures:
            for e in eligible_equipment[p, proc]:
                if (p, e) in x_vars and x_vars[p, e].X > 1e-06:
                    print(f'  {proc} on {e}: {x_vars[p, e].X:.2f} units')
    print('\n--- Equipment Utilization ---')
    for e in equipment_operating_time:
        used_time = sum((processing_time[p, e] * x_vars[p, e].X for p in products for proc in procedures if e in eligible_equipment[p, proc] and (p, e) in x_vars))
        print(f'{e}: Used {used_time:.2f} / {equipment_operating_time[e]:.2f} hours, Cost: {equipment_full_load_cost[e] * (used_time / equipment_operating_time[e]):.2f} yuan')
else:
    print(f'No optimal solution found. Status: {m.status}')