import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv'
df = pd.read_csv(csv_path, sep=',')
products = ['Product I', 'Product II', 'Product III']
product_short = {'Product I': 'I', 'Product II': 'II', 'Product III': 'III'}
procedures = ['A', 'B']
equip_rows = df['Equipment / Cost'].str.strip()
A_equip = [row for row in equip_rows if re.fullmatch('A\\d+', row)]
B_equip = [row for row in equip_rows if re.fullmatch('B\\d+', row)]
equip_proc = {}
for e in A_equip:
    equip_proc[e] = 'A'
for e in B_equip:
    equip_proc[e] = 'B'
eligibility = []
for p in products:
    if p == 'Product I':
        for e in ['A1', 'A2']:
            eligibility.append((p, 'A', e))
    elif p == 'Product II':
        for e in ['A1', 'A2']:
            eligibility.append((p, 'A', e))
    elif p == 'Product III':
        eligibility.append((p, 'A', 'A2'))
for p in products:
    if p == 'Product I':
        for e in ['B1', 'B2', 'B3']:
            eligibility.append((p, 'B', e))
    elif p == 'Product II':
        eligibility.append((p, 'B', 'B1'))
    elif p == 'Product III':
        eligibility.append((p, 'B', 'B2'))
equip_rows_mask = equip_rows.str.match('[AB]\\d+')
equip_df = df.loc[equip_rows_mask].copy()
equip_df['Equipment / Cost'] = equip_df['Equipment / Cost'].str.strip()
equip_df.set_index('Equipment / Cost', inplace=True)
avail_time = {}
equip_full_cost = {}
for e in equip_df.index:
    val = equip_df.at[e, 'Available Equipment Operating Time']
    if pd.isnull(val):
        raise ValueError(f'Missing available operating time for equipment {e}')
    avail_time[e] = float(val)
    val2 = equip_df.at[e, 'Equipment Cost at Full Load (yuan)']
    if pd.isnull(val2):
        raise ValueError(f'Missing equipment cost at full load for equipment {e}')
    equip_full_cost[e] = float(val2)
proc_time = {}
for e in equip_df.index:
    for p in products:
        val = equip_df.at[e, p]
        if pd.isnull(val):
            continue
        proc_time[p, e] = float(val)

def find_row_idx(label):
    mask = df['Equipment / Cost'].str.strip().str.casefold() == label.casefold()
    idxs = df.index[mask]
    if len(idxs) != 1:
        raise ValueError(f"Could not find unique row for '{label}'")
    return idxs[0]
raw_mat_idx = find_row_idx('Raw Material Cost (yuan/unit)')
unit_price_idx = find_row_idx('Unit Price (yuan/unit)')
raw_mat_cost = {}
unit_price = {}
for p in products:
    val = df.at[raw_mat_idx, p]
    if pd.isnull(val):
        raise ValueError(f'Missing raw material cost for {p}')
    raw_mat_cost[p] = float(val)
    val2 = df.at[unit_price_idx, p]
    if pd.isnull(val2):
        raise ValueError(f'Missing unit price for {p}')
    unit_price[p] = float(val2)
for (p, proc, e) in eligibility:
    if (p, e) not in proc_time:
        raise ValueError(f'Missing processing time for ({p}, {e})')

def solve_problem():
    m = gp.Model('FactoryProduction')
    m.Params.MIPGap = 0.0001
    x_keys = eligibility
    x = m.addVars(x_keys, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    for p in products:
        sum_A = gp.quicksum((x[p, 'A', e] for (pp, proc, e) in x_keys if pp == p and proc == 'A'))
        sum_B = gp.quicksum((x[p, 'B', e] for (pp, proc, e) in x_keys if pp == p and proc == 'B'))
        m.addConstr(sum_A == sum_B, name=f'sync_{product_short[p]}')
    for e in equip_df.index:
        expr = gp.LinExpr()
        for (p, proc, ee) in x_keys:
            if ee == e:
                expr += proc_time[p, e] * x[p, proc, e]
        m.addConstr(expr <= avail_time[e], name=f'cap_{e}')
    total_prod = {}
    for p in products:
        total_prod[p] = gp.quicksum((x[p, 'A', e] for (pp, proc, e) in x_keys if pp == p and proc == 'A'))
    revenue = gp.quicksum((unit_price[p] * total_prod[p] for p in products))
    raw_cost = gp.quicksum((raw_mat_cost[p] * total_prod[p] for p in products))
    equip_cost = gp.LinExpr()
    for e in equip_df.index:
        used_time = gp.quicksum((proc_time[p, e] * x[p, proc, e] for (p, proc, ee) in x_keys if ee == e))
        equip_cost += equip_full_cost[e] * (used_time / avail_time[e])
    m.setObjective(revenue - raw_cost - equip_cost, gp.GRB.MAXIMIZE)
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')