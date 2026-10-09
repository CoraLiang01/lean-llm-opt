import pandas as pd
import numpy as np
import re
from gurobipy import Model, GRB, quicksum
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv'
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
products = ['Product I', 'Product II', 'Product III']
procedure_A_equip = ['A1', 'A2']
procedure_B_equip = ['B1', 'B2', 'B3']
equipment_rows = df['Equipment / Cost'].isin(procedure_A_equip + procedure_B_equip)
equip_df = df[equipment_rows].copy()
equip_type = {}
for e in procedure_A_equip:
    equip_type[e] = 'A'
for e in procedure_B_equip:
    equip_type[e] = 'B'
equip_oper_time = {}
equip_cost_full_load = {}
equip_full_load_time = {}
for (idx, row) in equip_df.iterrows():
    e = row['Equipment / Cost'].strip()
    try:
        avail_time = float(row['Available Equipment Operating Time'])
    except ValueError:
        raise ValueError(f'Missing or invalid available operating time for equipment {e}')
    equip_oper_time[e] = avail_time
    try:
        cost_full = float(row['Equipment Cost at Full Load (yuan)'])
    except ValueError:
        raise ValueError(f'Missing or invalid equipment cost at full load for equipment {e}')
    equip_cost_full_load[e] = cost_full
    equip_full_load_time[e] = avail_time
proc_time = {}
for (idx, row) in equip_df.iterrows():
    e = row['Equipment / Cost'].strip()
    for p in products:
        val = row[p].strip()
        if val != '':
            try:
                proc_time[p, e] = float(val)
            except ValueError:
                raise ValueError(f'Invalid processing time for {p} on {e}: {val}')

def find_row_idx_by_prefix(prefix):
    for (idx, val) in df['Equipment / Cost'].items():
        if val.strip().casefold().startswith(prefix.casefold()):
            return idx
    raise ValueError(f"Row with prefix '{prefix}' not found.")
raw_mat_row = df.loc[find_row_idx_by_prefix('Raw Material Cost')]
unit_price_row = df.loc[find_row_idx_by_prefix('Unit Price')]
raw_mat_cost = {}
unit_price = {}
for p in products:
    try:
        raw_mat_cost[p] = float(raw_mat_row[p].strip())
    except ValueError:
        raise ValueError(f'Missing or invalid raw material cost for {p}')
    try:
        unit_price[p] = float(unit_price_row[p].strip())
    except ValueError:
        raise ValueError(f'Missing or invalid unit price for {p}')
A_compat = []
for p in products:
    if p == 'Product I' or p == 'Product II':
        for e in procedure_A_equip:
            if (p, e) in proc_time:
                A_compat.append((p, e))
    elif p == 'Product III':
        e = 'A2'
        if (p, e) in proc_time:
            A_compat.append((p, e))
B_compat = []
for p in products:
    if p == 'Product I':
        for e in procedure_B_equip:
            if (p, e) in proc_time:
                B_compat.append((p, e))
    elif p == 'Product II':
        e = 'B1'
        if (p, e) in proc_time:
            B_compat.append((p, e))
    elif p == 'Product III':
        e = 'B2'
        if (p, e) in proc_time:
            B_compat.append((p, e))
m = Model('factory_mixture3')
q_vars = m.addVars(products, lb=0, vtype=GRB.CONTINUOUS, name='')
a_vars = m.addVars(A_compat, lb=0, vtype=GRB.CONTINUOUS, name='')
b_vars = m.addVars(B_compat, lb=0, vtype=GRB.CONTINUOUS, name='')
for p in products:
    compat_A = [e for (pp, e) in A_compat if pp == p]
    if compat_A:
        m.addConstr(quicksum((a_vars[p, e] for e in compat_A)) == q_vars[p], name=f'procA_bal_{p}')
    else:
        m.addConstr(q_vars[p] == 0, name=f'procA_bal_{p}_zero')
    compat_B = [e for (pp, e) in B_compat if pp == p]
    if compat_B:
        m.addConstr(quicksum((b_vars[p, e] for e in compat_B)) == q_vars[p], name=f'procB_bal_{p}')
    else:
        m.addConstr(q_vars[p] == 0, name=f'procB_bal_{p}_zero')
for e in procedure_A_equip + procedure_B_equip:
    if e in procedure_A_equip:
        compat = [(p, e) for (p, ee) in A_compat if ee == e]
        var_dict = a_vars
    else:
        compat = [(p, e) for (p, ee) in B_compat if ee == e]
        var_dict = b_vars
    if compat:
        m.addConstr(quicksum((proc_time[p, e] * var_dict[p, e] for (p, e) in compat)) <= equip_oper_time[e], name=f'equip_time_{e}')
revenue = quicksum((unit_price[p] * q_vars[p] for p in products))
raw_cost = quicksum((raw_mat_cost[p] * q_vars[p] for p in products))
equip_cost_terms = []
for e in procedure_A_equip + procedure_B_equip:
    if e in procedure_A_equip:
        compat = [(p, e) for (p, ee) in A_compat if ee == e]
        var_dict = a_vars
    else:
        compat = [(p, e) for (p, ee) in B_compat if ee == e]
        var_dict = b_vars
    if compat:
        total_time_used = quicksum((proc_time[p, e] * var_dict[p, e] for (p, e) in compat))
        cost = total_time_used / equip_full_load_time[e] * equip_cost_full_load[e]
        equip_cost_terms.append(cost)
equip_cost = quicksum(equip_cost_terms)
m.setObjective(revenue - raw_cost - equip_cost, GRB.MAXIMIZE)
m.optimize()