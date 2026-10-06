import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv'
df = pd.read_csv(csv_path, sep=',')
products = ['Product I', 'Product II', 'Product III']
product_short = {'Product I': 'I', 'Product II': 'II', 'Product III': 'III'}
procedures = ['A', 'B']
equip_A = ['A1', 'A2']
equip_B = ['B1', 'B2', 'B3']
equip_all = equip_A + equip_B
compat = {('Product I', 'A'): ['A1', 'A2'], ('Product I', 'B'): ['B1', 'B2', 'B3'], ('Product II', 'A'): ['A1', 'A2'], ('Product II', 'B'): ['B1'], ('Product III', 'A'): ['A2'], ('Product III', 'B'): ['B2']}

def find_row_idx(label):
    idx = df['Equipment / Cost'].str.casefold().str.strip() == label.casefold().strip()
    matches = df[idx]
    if matches.empty:
        raise KeyError(f"Row '{label}' not found in CSV.")
    return matches.index[0]
proc_time = {}
for eq in equip_all:
    idx = find_row_idx(eq)
    for p in products:
        val = df.at[idx, p]
        if not (pd.isna(val) or str(val).strip() == ''):
            proc_time[p, eq] = float(val)
equip_time = {}
equip_cost = {}
for eq in equip_all:
    idx = find_row_idx(eq)
    avail_time = df.at[idx, 'Available Equipment Operating Time']
    cost_full = df.at[idx, 'Equipment Cost at Full Load (yuan)']
    if pd.isna(avail_time) or pd.isna(cost_full):
        raise ValueError(f'Missing available time or cost for equipment {eq}')
    equip_time[eq] = float(avail_time)
    equip_cost[eq] = float(cost_full)
idx_rm = find_row_idx('Raw Material Cost (yuan/unit)')
raw_mat_cost = {}
for p in products:
    val = df.at[idx_rm, p]
    if pd.isna(val) or str(val).strip() == '':
        raise ValueError(f'Missing raw material cost for {p}')
    raw_mat_cost[p] = float(val)
idx_price = find_row_idx('Unit Price (yuan/unit)')
unit_price = {}
for p in products:
    val = df.at[idx_price, p]
    if pd.isna(val) or str(val).strip() == '':
        raise ValueError(f'Missing unit price for {p}')
    unit_price[p] = float(val)
var_keys = []
A_equip_by_product = {}
B_equip_by_product = {}
for p in products:
    A_equip = compat.get((p, 'A'), [])
    A_equip_by_product[p] = A_equip
    for eq in A_equip:
        var_keys.append((p, 'A', eq))
    B_equip = compat.get((p, 'B'), [])
    B_equip_by_product[p] = B_equip
    for eq in B_equip:
        var_keys.append((p, 'B', eq))

def solve_problem():
    m = gp.Model('FactoryProduction')
    m.Params.MIPGap = 0.0001
    x = m.addVars(var_keys, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    for eq in equip_all:
        expr = gp.LinExpr()
        for (p, proc, e) in var_keys:
            if e == eq:
                expr += proc_time[p, eq] * x[p, proc, eq]
        m.addConstr(expr <= equip_time[eq], name=f'time_{eq}')
    for p in products:
        expr_A = gp.LinExpr()
        for eq in A_equip_by_product[p]:
            expr_A += x[p, 'A', eq]
        expr_B = gp.LinExpr()
        for eq in B_equip_by_product[p]:
            expr_B += x[p, 'B', eq]
        m.addConstr(expr_A == expr_B, name=f'consist_{product_short[p]}')
    total_prod = {}
    for p in products:
        total_prod[p] = gp.quicksum((x[p, 'A', eq] for eq in A_equip_by_product[p]))
    revenue = gp.quicksum((unit_price[p] * total_prod[p] for p in products))
    raw_cost = gp.quicksum((raw_mat_cost[p] * total_prod[p] for p in products))
    equip_cost_expr = gp.LinExpr()
    for eq in equip_all:
        used_time = gp.LinExpr()
        for (p, proc, e) in var_keys:
            if e == eq:
                used_time += proc_time[p, eq] * x[p, proc, eq]
        equip_cost_expr += equip_cost[eq] * (used_time / equip_time[eq])
    m.setObjective(revenue - raw_cost - equip_cost_expr, gp.GRB.MAXIMIZE)
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal:.6f}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X:.6f}')
else:
    print(f'Solver status: {m.status}')