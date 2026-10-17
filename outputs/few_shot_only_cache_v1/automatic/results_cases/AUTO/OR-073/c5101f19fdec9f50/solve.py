import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
products = ['Product I', 'Product II', 'Product III']
procedures = ['A', 'B']
A_equipment = ['A1', 'A2']
B_equipment = ['B1', 'B2', 'B3']
eligible_tuples = []
for p in products:
    if p == 'Product I' or p == 'Product II':
        for e in A_equipment:
            eligible_tuples.append((p, 'A', e))
    elif p == 'Product III':
        eligible_tuples.append((p, 'A', 'A2'))
for p in products:
    if p == 'Product I':
        for e in B_equipment:
            eligible_tuples.append((p, 'B', e))
    elif p == 'Product II':
        eligible_tuples.append((p, 'B', 'B1'))
    elif p == 'Product III':
        eligible_tuples.append((p, 'B', 'B2'))

def get_num(val):
    val = str(val).strip()
    return float(val) if val != '' else None
equip_rows = {}
for idx, row in df.iterrows():
    equip = row['Equipment / Cost'].strip()
    if equip in ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'B4']:
        equip_rows[equip] = idx
product_cols = {p: p for p in products}
proc_time = {}
for p, proc, e in eligible_tuples:
    if e not in equip_rows:
        raise ValueError(f'Equipment {e} not found in data')
    row = df.loc[equip_rows[e]]
    val = row[p]
    t = get_num(val)
    if t is None:
        raise ValueError(f'Missing processing time for {p} on {e}')
    proc_time[p, e] = t
equip_oper_time = {}
equip_full_cost = {}
for e in set((e for _, _, e in eligible_tuples)):
    if e not in equip_rows:
        raise ValueError(f'Equipment {e} not found in data')
    row = df.loc[equip_rows[e]]
    avail_time = get_num(row['Available Equipment Operating Time'])
    full_cost = get_num(row['Equipment Cost at Full Load (yuan)'])
    if avail_time is None or full_cost is None:
        raise ValueError(f'Missing available time or cost for equipment {e}')
    equip_oper_time[e] = avail_time
    equip_full_cost[e] = full_cost
raw_mat_row = df[df['Equipment / Cost'].str.strip() == 'Raw Material Cost (yuan/unit)']
if raw_mat_row.shape[0] != 1:
    raise ValueError('Raw Material Cost row not found or not unique')
raw_mat_row = raw_mat_row.iloc[0]
raw_mat_cost = {}
for p in products:
    val = get_num(raw_mat_row[p])
    if val is None:
        raise ValueError(f'Missing raw material cost for {p}')
    raw_mat_cost[p] = val
unit_price_row = df[df['Equipment / Cost'].str.strip() == 'Unit Price (yuan/unit)']
if unit_price_row.shape[0] != 1:
    raise ValueError('Unit Price row not found or not unique')
unit_price_row = unit_price_row.iloc[0]
unit_price = {}
for p in products:
    val = get_num(unit_price_row[p])
    if val is None:
        raise ValueError(f'Missing unit price for {p}')
    unit_price[p] = val
m = gp.Model('ProductionPlan')
x_vars = m.addVars(eligible_tuples, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
for e in equip_oper_time:
    relevant = [(p, proc, eq) for p, proc, eq in eligible_tuples if eq == e]
    m.addConstr(gp.quicksum((proc_time[p, e] * x_vars[p, proc, e] for p, proc, e in relevant)) <= equip_oper_time[e], name=f'equip_time_{e}')
for p in products:
    A_eqs = [e for pp, proc, e in eligible_tuples if pp == p and proc == 'A']
    B_eqs = [e for pp, proc, e in eligible_tuples if pp == p and proc == 'B']
    if not A_eqs or not B_eqs:
        raise ValueError(f'Product {p} missing eligible equipment for A or B')
    m.addConstr(gp.quicksum((x_vars[p, 'A', e] for e in A_eqs)) == gp.quicksum((x_vars[p, 'B', e] for e in B_eqs)), name=f'flow_{p}')
prod_total = {}
for p in products:
    B_eqs = [e for pp, proc, e in eligible_tuples if pp == p and proc == 'B']
    prod_total[p] = gp.quicksum((x_vars[p, 'B', e] for e in B_eqs))
equip_time_used = {}
for e in equip_oper_time:
    relevant = [(p, proc, eq) for p, proc, eq in eligible_tuples if eq == e]
    equip_time_used[e] = gp.quicksum((proc_time[p, e] * x_vars[p, proc, e] for p, proc, e in relevant))
total_revenue = gp.quicksum((unit_price[p] * prod_total[p] for p in products))
total_raw_mat = gp.quicksum((raw_mat_cost[p] * prod_total[p] for p in products))
total_equip_cost = gp.quicksum((equip_full_cost[e] * (equip_time_used[e] / equip_oper_time[e]) for e in equip_oper_time))
m.setObjective(total_revenue - total_raw_mat - total_equip_cost, gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total profit: {m.objVal:.4f}\n')
    print('Production plan (units processed on each equipment for each procedure):')
    for p, proc, e in eligible_tuples:
        val = x_vars[p, proc, e].X
        if val > 1e-06:
            print(f'  {p}, Procedure {proc}, Equipment {e}: {val:.4f} units')
    print('\nTotal units produced per product:')
    for p in products:
        print(f'  {p}: {prod_total[p].getValue():.4f} units')
    print('\nEquipment utilization and cost:')
    for e in equip_oper_time:
        used = equip_time_used[e].getValue()
        util = used / equip_oper_time[e]
        cost = equip_full_cost[e] * util
        print(f'  {e}: Used {used:.2f} / {equip_oper_time[e]:.2f} hours ({util * 100:.2f}%), Cost: {cost:.2f} yuan')
else:
    print(f'No optimal solution found. Status: {m.status}')