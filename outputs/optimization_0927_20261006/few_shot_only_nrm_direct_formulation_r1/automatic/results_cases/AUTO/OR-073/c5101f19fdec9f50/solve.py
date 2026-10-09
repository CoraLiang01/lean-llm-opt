import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
df.columns = [c.strip() for c in df.columns]
equipment_ids = ['A1', 'A2', 'B1', 'B2', 'B3']
equipment_rows = df[df['Equipment / Cost'].str.strip().isin(equipment_ids)].copy()
raw_material_row = df[df['Equipment / Cost'].str.strip().str.casefold() == 'raw material cost (yuan/unit)']
unit_price_row = df[df['Equipment / Cost'].str.strip().str.casefold() == 'unit price (yuan/unit)']
if equipment_rows.shape[0] != 5:
    raise ValueError('Did not find all 5 equipment rows in the CSV.')
if raw_material_row.shape[0] != 1:
    raise ValueError('Did not find the raw material cost row in the CSV.')
if unit_price_row.shape[0] != 1:
    raise ValueError('Did not find the unit price row in the CSV.')
products = ['Product I', 'Product II', 'Product III']
product_short = {'Product I': 'I', 'Product II': 'II', 'Product III': 'III'}
equipments = equipment_ids
processing_time = {}
for (_, row) in equipment_rows.iterrows():
    eq = row['Equipment / Cost'].strip()
    processing_time[eq] = {}
    for p in products:
        val = row[p].strip()
        if val == '':
            processing_time[eq][p] = None
        else:
            try:
                processing_time[eq][p] = float(val)
            except Exception:
                raise ValueError(f'Invalid processing time for equipment {eq}, product {p}: {val}')
available_time = {}
for (_, row) in equipment_rows.iterrows():
    eq = row['Equipment / Cost'].strip()
    val = row['Available Equipment Operating Time'].strip()
    try:
        available_time[eq] = float(val)
    except Exception:
        raise ValueError(f'Invalid available equipment operating time for {eq}: {val}')
equipment_cost_full = {}
for (_, row) in equipment_rows.iterrows():
    eq = row['Equipment / Cost'].strip()
    val = row['Equipment Cost at Full Load (yuan)'].strip()
    try:
        equipment_cost_full[eq] = float(val)
    except Exception:
        raise ValueError(f'Invalid equipment cost at full load for {eq}: {val}')
raw_material_cost = {}
for p in products:
    val = raw_material_row.iloc[0][p].strip()
    try:
        raw_material_cost[p] = float(val)
    except Exception:
        raise ValueError(f'Invalid raw material cost for {p}: {val}')
selling_price = {}
for p in products:
    val = unit_price_row.iloc[0][p].strip()
    try:
        selling_price[p] = float(val)
    except Exception:
        raise ValueError(f'Invalid unit price for {p}: {val}')
procedure_of_equipment = {}
for eq in equipments:
    if eq.startswith('A'):
        procedure_of_equipment[eq] = 'A'
    elif eq.startswith('B'):
        procedure_of_equipment[eq] = 'B'
    else:
        raise ValueError(f'Unknown equipment type: {eq}')
eligible_pairs = []
for p in products:
    if p == 'Product I':
        for eq in ['A1', 'A2']:
            eligible_pairs.append((p, eq))
        for eq in ['B1', 'B2', 'B3']:
            eligible_pairs.append((p, eq))
    elif p == 'Product II':
        for eq in ['A1', 'A2']:
            eligible_pairs.append((p, eq))
        eligible_pairs.append((p, 'B1'))
    elif p == 'Product III':
        eligible_pairs.append((p, 'A2'))
        eligible_pairs.append((p, 'B2'))
    else:
        raise ValueError(f'Unknown product: {p}')
m = gp.Model('ProductionPlan')
x_vars = m.addVars(eligible_pairs, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
for eq in equipments:
    eligible_ps = [p for (p, e) in eligible_pairs if e == eq]
    if not eligible_ps:
        continue
    m.addConstr(gp.quicksum((processing_time[eq][p] * x_vars[p, eq] for p in eligible_ps)) <= available_time[eq], name=f'time_{eq}')
for p in products:
    A_eqs = [eq for eq in equipments if procedure_of_equipment[eq] == 'A' and (p, eq) in eligible_pairs]
    B_eqs = [eq for eq in equipments if procedure_of_equipment[eq] == 'B' and (p, eq) in eligible_pairs]
    m.addConstr(gp.quicksum((x_vars[p, eq] for eq in A_eqs)) == gp.quicksum((x_vars[p, eq] for eq in B_eqs)), name=f'proc_balance_{product_short[p]}')
total_units = {}
for p in products:
    A_eqs = [eq for eq in equipments if procedure_of_equipment[eq] == 'A' and (p, eq) in eligible_pairs]
    total_units[p] = gp.quicksum((x_vars[p, eq] for eq in A_eqs))
total_revenue = gp.quicksum((selling_price[p] * total_units[p] for p in products))
total_raw_material_cost = gp.quicksum((raw_material_cost[p] * total_units[p] for p in products))
equipment_cost_exprs = []
for eq in equipments:
    eligible_ps = [p for (p, e) in eligible_pairs if e == eq]
    if not eligible_ps:
        continue
    total_time_used = gp.quicksum((processing_time[eq][p] * x_vars[p, eq] for p in eligible_ps))
    equipment_cost_exprs.append(equipment_cost_full[eq] * (total_time_used / available_time[eq]))
total_equipment_cost = gp.quicksum(equipment_cost_exprs)
m.setObjective(total_revenue - total_raw_material_cost - total_equipment_cost, gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total profit: {m.objVal:.2f}\n')
    print('--- Production Plan ---')
    for (p, eq) in eligible_pairs:
        val = x_vars[p, eq].X
        if val > 1e-06:
            print(f'  {p} on {eq}: {val:.2f} units')
    print('\n--- Total units produced per product ---')
    for p in products:
        units = total_units[p].getValue()
        print(f'  {p}: {units:.2f} units')
    print('\n--- Equipment utilization ---')
    for eq in equipments:
        eligible_ps = [p for (p, e) in eligible_pairs if e == eq]
        if not eligible_ps:
            continue
        total_time = sum((processing_time[eq][p] * x_vars[p, eq].X for p in eligible_ps))
        print(f'  {eq}: {total_time:.2f} / {available_time[eq]:.2f} hours used')
    print('\n--- Cost breakdown ---')
    print(f'  Total revenue: {total_revenue.getValue():.2f}')
    print(f'  Total raw material cost: {total_raw_material_cost.getValue():.2f}')
    print(f'  Total equipment cost: {total_equipment_cost.getValue():.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')