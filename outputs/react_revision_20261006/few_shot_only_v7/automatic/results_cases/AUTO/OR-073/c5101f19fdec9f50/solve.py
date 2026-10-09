import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
df.columns = [c.strip() for c in df.columns]
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
allowed_pairs = []
for e in ['A1', 'A2']:
    allowed_pairs.append(('Product I', e))
for e in ['B1', 'B2', 'B3']:
    allowed_pairs.append(('Product I', e))
for e in ['A1', 'A2']:
    allowed_pairs.append(('Product II', e))
allowed_pairs.append(('Product II', 'B1'))
allowed_pairs.append(('Product III', 'A2'))
allowed_pairs.append(('Product III', 'B2'))
allowed_pairs = [(p, e) for (p, e) in allowed_pairs]
processing_time = {}
available_time = {}
equipment_cost_full = {}
for (idx, row) in df.iterrows():
    equip = row['Equipment / Cost'].strip()
    if equip not in equipments:
        continue
    for p in products:
        val = row[p].strip()
        if val == '':
            continue
        try:
            pt = float(val)
        except ValueError:
            raise ValueError(f"Invalid processing time for {p} on {equip}: '{val}'")
        processing_time[p, equip] = pt
    atime_val = row['Available Equipment Operating Time'].strip()
    ecost_val = row['Equipment Cost at Full Load (yuan)'].strip()
    try:
        available_time[equip] = float(atime_val)
    except ValueError:
        raise ValueError(f"Invalid available time for {equip}: '{atime_val}'")
    try:
        equipment_cost_full[equip] = float(ecost_val)
    except ValueError:
        raise ValueError(f"Invalid equipment cost at full load for {equip}: '{ecost_val}'")
for (p, e) in allowed_pairs:
    if (p, e) not in processing_time:
        raise ValueError(f'Missing processing time for ({p}, {e}) in CSV.')

def find_param_row(label):
    mask = df['Equipment / Cost'].str.casefold() == label.casefold()
    if not mask.any():
        raise ValueError(f"Missing row for '{label}' in CSV.")
    return df.loc[mask].iloc[0]
selling_price_row = find_param_row('Selling Price')
raw_material_cost_row = find_param_row('Raw Material Cost')
selling_price = {}
raw_material_cost = {}
for p in products:
    sp_val = selling_price_row[p].strip()
    rc_val = raw_material_cost_row[p].strip()
    try:
        selling_price[p] = float(sp_val)
    except ValueError:
        raise ValueError(f"Invalid selling price for {p}: '{sp_val}'")
    try:
        raw_material_cost[p] = float(rc_val)
    except ValueError:
        raise ValueError(f"Invalid raw material cost for {p}: '{rc_val}'")

def solve_problem():
    m = gp.Model('FactoryProductionPlan')
    quantity_vars = m.addVars(allowed_pairs, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    product_A_equip = {'Product I': ['A1', 'A2'], 'Product II': ['A1', 'A2'], 'Product III': ['A2']}
    product_B_equip = {'Product I': ['B1', 'B2', 'B3'], 'Product II': ['B1'], 'Product III': ['B2']}
    for p in products:
        m.addConstr(gp.quicksum((quantity_vars[p, e] for e in product_A_equip[p] if (p, e) in quantity_vars)) == gp.quicksum((quantity_vars[p, e] for e in product_B_equip[p] if (p, e) in quantity_vars)), name=f'flow_{p}')
    for e in equipments:
        relevant_pairs = [(p, e) for p in products if (p, e) in quantity_vars]
        if not relevant_pairs:
            continue
        m.addConstr(gp.quicksum((processing_time[p, e] * quantity_vars[p, e] for (p, e) in relevant_pairs)) <= available_time[e], name=f'cap_{e}')
    total_revenue = gp.quicksum((selling_price[p] * gp.quicksum((quantity_vars[p, e] for e in product_A_equip[p] if (p, e) in quantity_vars)) for p in products))
    total_raw_cost = gp.quicksum((raw_material_cost[p] * gp.quicksum((quantity_vars[p, e] for e in product_A_equip[p] if (p, e) in quantity_vars)) for p in products))
    total_equipment_cost = gp.quicksum((equipment_cost_full[e] * (gp.quicksum((processing_time[p, e] * quantity_vars[p, e] for p in products if (p, e) in quantity_vars)) / available_time[e]) for e in equipments if available_time[e] > 0))
    m.setObjective(total_revenue - total_raw_cost - total_equipment_cost, gp.GRB.MAXIMIZE)
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