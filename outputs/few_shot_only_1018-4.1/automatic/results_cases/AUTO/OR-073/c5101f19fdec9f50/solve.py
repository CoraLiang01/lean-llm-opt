import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
products = ['Product I', 'Product II', 'Product III']
product_short = {'Product I': 'I', 'Product II': 'II', 'Product III': 'III'}
procedures = ['A', 'B']
equip_A = ['A1', 'A2']
equip_B = ['B1', 'B2', 'B3']

def norm_equipment(e):
    return e.strip().upper()
equipment_rows = []
for idx, row in df.iterrows():
    equip = row['Equipment / Cost'].strip()
    if re.match('^(A1|A2|B1|B2|B3)$', equip):
        equipment_rows.append(idx)
equipment_params = {}
for idx in equipment_rows:
    row = df.loc[idx]
    equip = row['Equipment / Cost'].strip()
    proc_times = {}
    for prod in products:
        val = row.get(prod, '').strip()
        proc_times[prod] = float(val) if val != '' else None
    avail_time = row.get('Available Equipment Operating Time', '').strip()
    avail_time = float(avail_time) if avail_time != '' else None
    equip_cost = row.get('Equipment Cost at Full Load (yuan)', '').strip()
    equip_cost = float(equip_cost) if equip_cost != '' else None
    equipment_params[equip] = {'processing_time': proc_times, 'available_time': avail_time, 'full_load_cost': equip_cost}

def find_row_idx_by_first_col(val):
    for idx, row in df.iterrows():
        if row['Equipment / Cost'].strip().casefold() == val.strip().casefold():
            return idx
    raise KeyError(f"Row with Equipment / Cost == '{val}' not found.")
raw_mat_row = df.loc[find_row_idx_by_first_col('Raw Material Cost (yuan/unit)')]
unit_price_row = df.loc[find_row_idx_by_first_col('Unit Price (yuan/unit)')]
raw_material_cost = {}
unit_price = {}
for prod in products:
    val_rm = raw_mat_row.get(prod, '').strip()
    raw_material_cost[prod] = float(val_rm) if val_rm != '' else None
    val_up = unit_price_row.get(prod, '').strip()
    unit_price[prod] = float(val_up) if val_up != '' else None
eligible_A = []
for prod in products:
    if prod == 'Product III':
        eligible_A.append(('Product III', 'A2'))
    else:
        for e in equip_A:
            eligible_A.append((prod, e))
eligible_B = []
for prod in products:
    if prod == 'Product I':
        for e in equip_B:
            eligible_B.append(('Product I', e))
    elif prod == 'Product II':
        eligible_B.append(('Product II', 'B1'))
    elif prod == 'Product III':
        eligible_B.append(('Product III', 'B2'))
x_vars = {}
m = gp.Model('ProductionPlan')
for prod, equip in eligible_A + eligible_B:
    varname = f'x_{product_short[prod]}_{equip}'
    x_vars[prod, equip] = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name=varname)
for equip in equip_A + equip_B:
    eligible_pairs = []
    for prod, eq in x_vars:
        if eq == equip:
            eligible_pairs.append((prod, eq))
    if not eligible_pairs:
        continue
    terms = []
    for prod, eq in eligible_pairs:
        pt = equipment_params[equip]['processing_time'][prod]
        if pt is None:
            raise ValueError(f'Missing processing time for {prod} on {equip}')
        terms.append(pt * x_vars[prod, eq])
    avail_time = equipment_params[equip]['available_time']
    if avail_time is None:
        raise ValueError(f'Missing available time for equipment {equip}')
    m.addConstr(gp.quicksum(terms) <= avail_time, name=f'time_{equip}')
for prod in products:
    sum_A = gp.quicksum((x_vars[p, e] for p, e in eligible_A if p == prod))
    sum_B = gp.quicksum((x_vars[p, e] for p, e in eligible_B if p == prod))
    m.addConstr(sum_A == sum_B, name=f'flow_{product_short[prod]}')
revenue_terms = []
for prod in products:
    total_prod = gp.quicksum((x_vars[p, e] for p, e in eligible_A if p == prod))
    up = unit_price[prod]
    if up is None:
        raise ValueError(f'Missing unit price for {prod}')
    revenue_terms.append(up * total_prod)
raw_mat_terms = []
for prod in products:
    total_prod = gp.quicksum((x_vars[p, e] for p, e in eligible_A if p == prod))
    rm = raw_material_cost[prod]
    if rm is None:
        raise ValueError(f'Missing raw material cost for {prod}')
    raw_mat_terms.append(rm * total_prod)
equip_cost_terms = []
for equip in equip_A + equip_B:
    time_used = gp.quicksum((equipment_params[equip]['processing_time'][prod] * x_vars[prod, equip] for prod, eq in x_vars if eq == equip))
    avail_time = equipment_params[equip]['available_time']
    full_load_cost = equipment_params[equip]['full_load_cost']
    if avail_time is None or full_load_cost is None:
        raise ValueError(f'Missing equipment parameters for {equip}')
    equip_cost_terms.append(full_load_cost * (time_used / avail_time))
m.setObjective(gp.quicksum(revenue_terms) - gp.quicksum(raw_mat_terms) - gp.quicksum(equip_cost_terms), gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total profit: {m.objVal:.4f}\n')
    print('Production quantities (units):')
    for (prod, equip), var in x_vars.items():
        if var.X > 1e-06:
            print(f'  {prod} on {equip}: {var.X:.4f}')
    print('\nTotal produced per product:')
    for prod in products:
        total = sum((x_vars[p, e].X for p, e in eligible_A if p == prod))
        print(f'  {prod}: {total:.4f}')
    print('\nEquipment utilization (hours used / available):')
    for equip in equip_A + equip_B:
        time_used = sum((equipment_params[equip]['processing_time'][prod] * x_vars[prod, equip].X for prod, eq in x_vars if eq == equip))
        avail_time = equipment_params[equip]['available_time']
        print(f'  {equip}: {time_used:.4f} / {avail_time}')
else:
    print(f'No optimal solution found. Status: {m.status}')