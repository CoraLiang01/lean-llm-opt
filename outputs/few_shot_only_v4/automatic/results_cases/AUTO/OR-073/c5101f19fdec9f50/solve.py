import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
products = ['I', 'II', 'III']
procedures = ['A', 'B']
equipments_A = ['A1', 'A2']
equipments_B = ['B1', 'B2', 'B3']
equipments = equipments_A + equipments_B
eligible_equipment = {('I', 'A'): ['A1', 'A2'], ('I', 'B'): ['B1', 'B2', 'B3'], ('II', 'A'): ['A1', 'A2'], ('II', 'B'): ['B1'], ('III', 'A'): ['A2'], ('III', 'B'): ['B2']}

def find_row_idx(df, pattern):
    norm = lambda s: str(s).strip().casefold()
    for idx, val in enumerate(df.iloc[:, 0]):
        if re.fullmatch(pattern, norm(val)):
            return idx
    raise KeyError(f"Row matching '{pattern}' not found.")

def find_col_idx(df, pattern):
    norm = lambda s: str(s).strip().casefold()
    for idx, val in enumerate(df.columns):
        if re.fullmatch(pattern, norm(val)):
            return idx
    raise KeyError(f"Column matching '{pattern}' not found.")

def norm(s):
    return str(s).strip().casefold()
col_labels = list(df.columns)
row_labels = list(df.iloc[:, 0])
product_col_map = {}
for p in products:
    for col in col_labels:
        if norm(col) == norm(p):
            product_col_map[p] = col
            break
    else:
        raise KeyError(f"Product column '{p}' not found in CSV.")
equipment_row_map = {}
for e in equipments:
    for idx, val in enumerate(row_labels):
        if norm(val) == norm(e):
            equipment_row_map[e] = idx
            break
    else:
        raise KeyError(f"Equipment row '{e}' not found in CSV.")
raw_material_row_idx = find_row_idx(df, 'raw material cost \\(yuan/unit\\)')
unit_price_row_idx = find_row_idx(df, 'unit price \\(yuan/unit\\)')
avail_time_col = None
equip_cost_col = None
for col in col_labels:
    if re.fullmatch('available equipment operating time', norm(col)):
        avail_time_col = col
    if re.fullmatch('equipment cost at full load \\(yuan\\)', norm(col)):
        equip_cost_col = col
if avail_time_col is None:
    raise KeyError("Column 'Available Equipment Operating Time' not found.")
if equip_cost_col is None:
    raise KeyError("Column 'Equipment Cost at Full Load (yuan)' not found.")
processing_time = {}
for e in equipments:
    row_idx = equipment_row_map[e]
    processing_time[e] = {}
    for p in products:
        col = product_col_map[p]
        val = df.at[row_idx, col]
        if val.strip() == '':
            processing_time[e][p] = None
        else:
            try:
                processing_time[e][p] = float(val)
            except Exception:
                raise ValueError(f"Invalid processing time for equipment {e}, product {p}: '{val}'")
raw_material_cost = {}
for p in products:
    col = product_col_map[p]
    val = df.at[raw_material_row_idx, col]
    try:
        raw_material_cost[p] = float(val)
    except Exception:
        raise ValueError(f"Invalid raw material cost for product {p}: '{val}'")
unit_price = {}
for p in products:
    col = product_col_map[p]
    val = df.at[unit_price_row_idx, col]
    try:
        unit_price[p] = float(val)
    except Exception:
        raise ValueError(f"Invalid unit price for product {p}: '{val}'")
available_time = {}
for e in equipments:
    row_idx = equipment_row_map[e]
    val = df.at[row_idx, avail_time_col]
    try:
        available_time[e] = float(val)
    except Exception:
        raise ValueError(f"Invalid available equipment operating time for equipment {e}: '{val}'")
equipment_cost_full = {}
for e in equipments:
    row_idx = equipment_row_map[e]
    val = df.at[row_idx, equip_cost_col]
    try:
        equipment_cost_full[e] = float(val)
    except Exception:
        raise ValueError(f"Invalid equipment cost at full load for equipment {e}: '{val}'")
A_pairs = []
B_pairs = []
for p in products:
    for e in eligible_equipment[p, 'A']:
        if processing_time[e][p] is not None:
            A_pairs.append((p, e))
    for e in eligible_equipment[p, 'B']:
        if processing_time[e][p] is not None:
            B_pairs.append((p, e))
equipment_pairs = {e: [] for e in equipments}
for p, e in A_pairs + B_pairs:
    equipment_pairs[e].append((p, e))
m = gp.Model('ProductionPlan')
x_A_vars = m.addVars(A_pairs, lb=0.0, name='')
x_B_vars = m.addVars(B_pairs, lb=0.0, name='')
for e in equipments:
    expr = gp.LinExpr()
    for p, eq in equipment_pairs[e]:
        if (p, eq) in A_pairs:
            expr += processing_time[e][p] * x_A_vars[p, e]
        if (p, eq) in B_pairs:
            expr += processing_time[e][p] * x_B_vars[p, e]
    m.addConstr(expr <= available_time[e], name=f'time_{e}')
for p in products:
    expr_A = gp.quicksum((x_A_vars[p, e] for e in eligible_equipment[p, 'A'] if (p, e) in x_A_vars))
    expr_B = gp.quicksum((x_B_vars[p, e] for e in eligible_equipment[p, 'B'] if (p, e) in x_B_vars))
    m.addConstr(expr_A == expr_B, name=f'proc_balance_{p}')
total_revenue = gp.quicksum((unit_price[p] * gp.quicksum((x_A_vars[p, e] for e in eligible_equipment[p, 'A'] if (p, e) in x_A_vars)) for p in products))
total_raw_material_cost = gp.quicksum((raw_material_cost[p] * gp.quicksum((x_A_vars[p, e] for e in eligible_equipment[p, 'A'] if (p, e) in x_A_vars)) for p in products))
equipment_cost_exprs = []
for e in equipments:
    total_time_used = gp.LinExpr()
    for p, eq in equipment_pairs[e]:
        if (p, eq) in A_pairs:
            total_time_used += processing_time[e][p] * x_A_vars[p, e]
        if (p, eq) in B_pairs:
            total_time_used += processing_time[e][p] * x_B_vars[p, e]
    if available_time[e] == 0:
        raise ValueError(f'Available equipment operating time for {e} is zero.')
    equipment_cost_exprs.append(equipment_cost_full[e] * (total_time_used / available_time[e]))
total_equipment_cost = gp.quicksum(equipment_cost_exprs)
m.setObjective(total_revenue - total_raw_material_cost - total_equipment_cost, gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total profit: {m.objVal:.2f}\n')
    print('Production Plan (quantities by procedure and equipment):')
    for p, e in x_A_vars:
        val = x_A_vars[p, e].X
        if val > 1e-06:
            print(f'  Procedure A: Product {p} on {e}: {val:.2f}')
    for p, e in x_B_vars:
        val = x_B_vars[p, e].X
        if val > 1e-06:
            print(f'  Procedure B: Product {p} on {e}: {val:.2f}')
    print('\nTotal produced per product:')
    for p in products:
        total = sum((x_A_vars[p, e].X for e in eligible_equipment[p, 'A'] if (p, e) in x_A_vars))
        print(f'  Product {p}: {total:.2f}')
    print('\nEquipment utilization (fraction of available time used):')
    for e in equipments:
        total_time = 0.0
        for p, eq in equipment_pairs[e]:
            if (p, eq) in A_pairs:
                total_time += processing_time[e][p] * x_A_vars[p, e].X
            if (p, eq) in B_pairs:
                total_time += processing_time[e][p] * x_B_vars[p, e].X
        print(f'  {e}: {total_time:.2f} / {available_time[e]:.2f} ({100 * total_time / available_time[e]:.1f}%)')
else:
    print(f'No optimal solution found. Status: {m.status}')