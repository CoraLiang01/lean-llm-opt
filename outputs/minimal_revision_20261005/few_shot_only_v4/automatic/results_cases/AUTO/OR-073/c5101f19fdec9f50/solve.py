import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv'
df = pd.read_csv(csv_path, sep=',')
products = ['Product I', 'Product II', 'Product III']
product_short = {'Product I': 'I', 'Product II': 'II', 'Product III': 'III'}
procedures = ['A', 'B']
equip_A = ['A1', 'A2']
equip_B = ['B1', 'B2', 'B3']
equip_all = equip_A + equip_B
equip_to_proc = {}
for e in equip_A:
    equip_to_proc[e] = 'A'
for e in equip_B:
    equip_to_proc[e] = 'B'
allowed_z = []
for p in products:
    if p == 'Product I':
        for e in equip_A:
            allowed_z.append((p, 'A', e))
        for e in equip_B:
            allowed_z.append((p, 'B', e))
    elif p == 'Product II':
        for e in equip_A:
            allowed_z.append((p, 'A', e))
        allowed_z.append((p, 'B', 'B1'))
    elif p == 'Product III':
        allowed_z.append((p, 'A', 'A2'))
        allowed_z.append((p, 'B', 'B2'))

def get_row(label):
    mask = df['Equipment / Cost'].str.casefold().str.replace(' ', '') == label.casefold().replace(' ', '')
    matches = df[mask]
    if matches.empty:
        raise KeyError(f"Row '{label}' not found in CSV.")
    return matches.iloc[0]
unit_price_row = get_row('Unit Price (yuan/unit)')
unit_price = {p: float(unit_price_row[p]) for p in products}
raw_mat_row = get_row('Raw Material Cost (yuan/unit)')
raw_mat_cost = {p: float(raw_mat_row[p]) for p in products}
equip_rows = {}
for e in equip_all:
    mask = df['Equipment / Cost'].str.casefold().str.replace(' ', '') == e.casefold().replace(' ', '')
    matches = df[mask]
    if matches.empty:
        raise KeyError(f"Equipment row '{e}' not found in CSV.")
    equip_rows[e] = matches.iloc[0]
proc_time = {}
for e in equip_all:
    for p in products:
        val = equip_rows[e][p]
        if pd.isnull(val):
            raise KeyError(f'Missing processing time for equipment {e}, product {p}.')
        proc_time[e, p] = float(val)
avail_time = {}
equip_cost_full = {}
for e in equip_all:
    row = equip_rows[e]
    atime = row['Available Equipment Operating Time']
    ecost = row['Equipment Cost at Full Load (yuan)']
    if pd.isnull(atime) or pd.isnull(ecost):
        raise KeyError(f'Missing available time or equipment cost for equipment {e}.')
    avail_time[e] = float(atime)
    equip_cost_full[e] = float(ecost)

def solve_problem():
    m = gp.Model('FactoryProductionPlan')
    x = m.addVars(products, lb=0, vtype=gp.GRB.CONTINUOUS, name='')
    z_keys = allowed_z
    z = m.addVars(z_keys, lb=0, vtype=gp.GRB.CONTINUOUS, name='')
    for p in products:
        equip_A_p = [e for (pp, s, e) in z_keys if pp == p and s == 'A']
        if equip_A_p:
            m.addConstr(gp.quicksum((z[p, 'A', e] for e in equip_A_p)) == x[p], name=f'assign_{p}_A')
        equip_B_p = [e for (pp, s, e) in z_keys if pp == p and s == 'B']
        if equip_B_p:
            m.addConstr(gp.quicksum((z[p, 'B', e] for e in equip_B_p)) == x[p], name=f'assign_{p}_B')
    for e in equip_all:
        z_e_keys = [(p, s, ee) for (p, s, ee) in z_keys if ee == e]
        expr = gp.quicksum((proc_time[e, p] * z[p, s, e] for (p, s, e) in z_e_keys))
        m.addConstr(expr <= avail_time[e], name=f'time_{e}')
    revenue = gp.quicksum((unit_price[p] * x[p] for p in products))
    raw_cost = gp.quicksum((raw_mat_cost[p] * x[p] for p in products))
    equip_cost_terms = []
    for e in equip_all:
        z_e_keys = [(p, s, ee) for (p, s, ee) in z_keys if ee == e]
        total_time_used = gp.quicksum((proc_time[e, p] * z[p, s, e] for (p, s, e) in z_e_keys))
        if avail_time[e] <= 0:
            raise ValueError(f'Available time for equipment {e} is zero or negative.')
        usage_frac = total_time_used / avail_time[e]
        equip_cost_terms.append(usage_frac * equip_cost_full[e])
    equip_cost = gp.quicksum(equip_cost_terms)
    m.setObjective(revenue - raw_cost - equip_cost, gp.GRB.MAXIMIZE)
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal:.6f}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X:.6f}')
else:
    print(f'Solver status: {m.status}')