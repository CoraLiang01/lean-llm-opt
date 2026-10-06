import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv'
df = pd.read_csv(csv_path, sep=',')
products = ['Product I', 'Product II', 'Product III']
product_short = {'Product I': 'I', 'Product II': 'II', 'Product III': 'III'}
equip_A = ['A1', 'A2']
equip_B = ['B1', 'B2', 'B3']
equip_rows = {}
for eq in equip_A + equip_B:
    matches = df['Equipment / Cost'].astype(str).str.strip().str.casefold() == eq.casefold()
    eq_idx = np.where(matches)[0]
    if len(eq_idx) != 1:
        raise ValueError(f"Equipment '{eq}' not found or not unique in CSV.")
    equip_rows[eq] = eq_idx[0]

def find_row_idx(label):
    matches = df['Equipment / Cost'].astype(str).str.strip().str.casefold() == label.casefold()
    idx = np.where(matches)[0]
    if len(idx) != 1:
        raise ValueError(f"Row '{label}' not found or not unique in CSV.")
    return idx[0]
raw_mat_row = find_row_idx('Raw Material Cost (yuan/unit)')
unit_price_row = find_row_idx('Unit Price (yuan/unit)')
raw_material_cost = {}
unit_price = {}
for p in products:
    val_rm = df.at[raw_mat_row, p]
    val_up = df.at[unit_price_row, p]
    if pd.isnull(val_rm) or pd.isnull(val_up):
        raise ValueError(f'Missing raw material cost or unit price for {p}')
    raw_material_cost[product_short[p]] = float(val_rm)
    unit_price[product_short[p]] = float(val_up)
available_time = {}
equip_cost_full = {}
for eq in equip_A + equip_B:
    row = equip_rows[eq]
    atime = df.at[row, 'Available Equipment Operating Time']
    ecost = df.at[row, 'Equipment Cost at Full Load (yuan)']
    if pd.isnull(atime) or pd.isnull(ecost):
        raise ValueError(f'Missing available time or equipment cost for {eq}')
    available_time[eq] = float(atime)
    equip_cost_full[eq] = float(ecost)
proc_time = {}
for eq in equip_A + equip_B:
    row = equip_rows[eq]
    for p in products:
        val = df.at[row, p]
        if not pd.isnull(val):
            proc_time[product_short[p], eq] = float(val)
compat_A = {'I': ['A1', 'A2'], 'II': ['A1', 'A2'], 'III': ['A2']}
compat_B = {'I': ['B1', 'B2', 'B3'], 'II': ['B1'], 'III': ['B2']}
pairs_A = [(p, e) for p in compat_A for e in compat_A[p]]
pairs_B = [(p, e) for p in compat_B for e in compat_B[p]]
m = gp.Model('FactoryProductionPlan')
x = {}
for p, e in pairs_A + pairs_B:
    x[p, e] = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name=f'x_{p}_{e}')
for eq in equip_A + equip_B:
    relevant_pairs = []
    if eq in equip_A:
        relevant_pairs = [(p, eq) for p in compat_A if eq in compat_A[p]]
    elif eq in equip_B:
        relevant_pairs = [(p, eq) for p in compat_B if eq in compat_B[p]]
    m.addConstr(gp.quicksum((proc_time[p, eq] * x[p, eq] for p, eq in relevant_pairs)) <= available_time[eq], name=f'EquipTime_{eq}')
for p in ['I', 'II', 'III']:
    sum_A = gp.quicksum((x[p, e] for e in compat_A[p]))
    sum_B = gp.quicksum((x[p, e] for e in compat_B[p]))
    m.addConstr(sum_A == sum_B, name=f'ProdConsist_{p}')
prod_total = {}
for p in ['I', 'II', 'III']:
    prod_total[p] = gp.quicksum((x[p, e] for e in compat_A[p]))
total_revenue = gp.quicksum((unit_price[p] * prod_total[p] for p in ['I', 'II', 'III']))
total_rawmat = gp.quicksum((raw_material_cost[p] * prod_total[p] for p in ['I', 'II', 'III']))
equip_cost_exprs = []
for eq in equip_A + equip_B:
    if eq in equip_A:
        relevant_pairs = [(p, eq) for p in compat_A if eq in compat_A[p]]
    else:
        relevant_pairs = [(p, eq) for p in compat_B if eq in compat_B[p]]
    used_time = gp.quicksum((proc_time[p, eq] * x[p, eq] for p, eq in relevant_pairs))
    equip_cost_exprs.append(equip_cost_full[eq] * (used_time / available_time[eq]))
total_equip_cost = gp.quicksum(equip_cost_exprs)
m.setObjective(total_revenue - total_rawmat - total_equip_cost, gp.GRB.MAXIMIZE)
m.optimize()