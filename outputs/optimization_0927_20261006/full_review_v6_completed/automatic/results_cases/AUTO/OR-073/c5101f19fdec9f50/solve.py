import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
products = ['Product I', 'Product II', 'Product III']
product_short = {'Product I': 'I', 'Product II': 'II', 'Product III': 'III'}
procedures = ['A', 'B']
equipment_rows = df[~df['Equipment / Cost'].str.strip().str.casefold().isin(['raw material cost (yuan/unit)', 'unit price (yuan/unit)']) & (df['Equipment / Cost'].str.strip() != '')].copy()
A_equipment = []
B_equipment = []
for eq in equipment_rows['Equipment / Cost']:
    eq_norm = eq.strip().upper()
    if eq_norm.startswith('A'):
        A_equipment.append(eq.strip())
    elif eq_norm.startswith('B'):
        B_equipment.append(eq.strip())
equipment = A_equipment + B_equipment
processing_time = {}
for (_, row) in equipment_rows.iterrows():
    eq = row['Equipment / Cost'].strip()
    for prod in products:
        val = row[prod].strip()
        if val != '':
            processing_time[prod, eq] = float(val)
available_time = {}
full_load_cost = {}
for (_, row) in equipment_rows.iterrows():
    eq = row['Equipment / Cost'].strip()
    atime = row['Available Equipment Operating Time'].strip()
    acost = row['Equipment Cost at Full Load (yuan)'].strip()
    if atime != '':
        available_time[eq] = float(atime)
    if acost != '':
        full_load_cost[eq] = float(acost)

def find_row_idx_by_name(name):
    mask = df['Equipment / Cost'].str.strip().str.casefold() == name.strip().casefold()
    idxs = df.index[mask]
    if len(idxs) == 0:
        raise KeyError(f"Row '{name}' not found in CSV.")
    return idxs[0]
raw_material_cost_row = find_row_idx_by_name('Raw Material Cost (yuan/unit)')
unit_price_row = find_row_idx_by_name('Unit Price (yuan/unit)')
raw_material_cost = {}
unit_price = {}
for prod in products:
    val_rm = df.at[raw_material_cost_row, prod].strip()
    val_up = df.at[unit_price_row, prod].strip()
    if val_rm != '':
        raw_material_cost[prod] = float(val_rm)
    if val_up != '':
        unit_price[prod] = float(val_up)
eligibility = {}
for prod in products:
    for eq in equipment:
        if eq in A_equipment:
            if prod == 'Product I' and eq in ['A1', 'A2']:
                eligibility[prod, eq, 'A'] = True
            elif prod == 'Product II' and eq in ['A1', 'A2']:
                eligibility[prod, eq, 'A'] = True
            elif prod == 'Product III' and eq == 'A2':
                eligibility[prod, eq, 'A'] = True
            else:
                eligibility[prod, eq, 'A'] = False
        elif eq in B_equipment:
            if prod == 'Product I' and eq in ['B1', 'B2', 'B3']:
                eligibility[prod, eq, 'B'] = True
            elif prod == 'Product II' and eq == 'B1':
                eligibility[prod, eq, 'B'] = True
            elif prod == 'Product III' and eq == 'B2':
                eligibility[prod, eq, 'B'] = True
            else:
                eligibility[prod, eq, 'B'] = False
m = gp.Model('FactoryProductionPlan')
x_vars = {}
for prod in products:
    for eq in equipment:
        for proc in procedures:
            if eligibility.get((prod, eq, proc), False):
                x_vars[prod, eq, proc] = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name=f'x_{product_short[prod]}_{eq}_{proc}')
for prod in products:
    sum_A = gp.quicksum((x_vars[prod, eq, 'A'] for eq in A_equipment if (prod, eq, 'A') in x_vars))
    sum_B = gp.quicksum((x_vars[prod, eq, 'B'] for eq in B_equipment if (prod, eq, 'B') in x_vars))
    m.addConstr(sum_A == sum_B, name=f'balance_{product_short[prod]}')
for eq in equipment:
    if eq in A_equipment:
        proc = 'A'
    elif eq in B_equipment:
        proc = 'B'
    else:
        continue
    total_time = gp.quicksum((processing_time[prod, eq] * x_vars[prod, eq, proc] for prod in products if (prod, eq, proc) in x_vars))
    if eq in available_time:
        m.addConstr(total_time <= available_time[eq], name=f'time_{eq}')
product_total = {}
for prod in products:
    product_total[prod] = gp.quicksum((x_vars[prod, eq, 'A'] for eq in A_equipment if (prod, eq, 'A') in x_vars))
total_revenue = gp.quicksum((unit_price[prod] * product_total[prod] for prod in products))
total_raw_material_cost = gp.quicksum((raw_material_cost[prod] * product_total[prod] for prod in products))
total_equipment_cost = gp.quicksum((full_load_cost[eq] * (gp.quicksum((processing_time[prod, eq] * x_vars[prod, eq, 'A' if eq in A_equipment else 'B'] for prod in products if (prod, eq, 'A' if eq in A_equipment else 'B') in x_vars)) / available_time[eq]) for eq in equipment if eq in available_time and eq in full_load_cost))
m.setObjective(total_revenue - total_raw_material_cost - total_equipment_cost, gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total profit: {m.objVal:.2f}')
    print('\n--- Production Plan ---')
    for prod in products:
        qty = product_total[prod].getValue()
        print(f'Product {product_short[prod]}: {qty:.2f} units')
        for eq in A_equipment:
            key = (prod, eq, 'A')
            if key in x_vars and x_vars[key].X > 1e-06:
                print(f'  Procedure A on {eq}: {x_vars[key].X:.2f} units')
        for eq in B_equipment:
            key = (prod, eq, 'B')
            if key in x_vars and x_vars[key].X > 1e-06:
                print(f'  Procedure B on {eq}: {x_vars[key].X:.2f} units')
    print('\n--- Equipment Utilization ---')
    for eq in equipment:
        if eq in A_equipment:
            proc = 'A'
        else:
            proc = 'B'
        used_time = sum((processing_time[prod, eq] * x_vars[prod, eq, proc].X for prod in products if (prod, eq, proc) in x_vars))
        print(f'{eq}: Used {used_time:.2f} / {available_time[eq]:.2f} hours')
else:
    print(f'No optimal solution found. Status: {m.status}')