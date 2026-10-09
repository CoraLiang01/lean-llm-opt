import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv'
df = pd.read_csv(csv_path, sep=',')
products = ['Product I', 'Product II', 'Product III']
equipments = ['A1', 'A2', 'B1', 'B2', 'B3']
equip_proc = {'A1': 'A', 'A2': 'A', 'B1': 'B', 'B2': 'B', 'B3': 'B'}
feasible_pairs = []
feasible_pairs += [('Product I', 'A1'), ('Product I', 'A2')]
feasible_pairs += [('Product II', 'A1'), ('Product II', 'A2')]
feasible_pairs += [('Product III', 'A2')]
feasible_pairs += [('Product I', 'B1'), ('Product I', 'B2'), ('Product I', 'B3')]
feasible_pairs += [('Product II', 'B1')]
feasible_pairs += [('Product III', 'B2')]

def row_idx(label):
    idx = df[df['Equipment / Cost'].astype(str).str.strip() == label].index
    if len(idx) == 0:
        raise KeyError(f"Row '{label}' not found in CSV.")
    return idx[0]
proc_time = {}
for e in equipments:
    if e not in df['Equipment / Cost'].values:
        raise KeyError(f"Equipment '{e}' not found in CSV.")
    row = df[df['Equipment / Cost'] == e].iloc[0]
    for p in products:
        if (p, e) in feasible_pairs:
            val = row[p]
            if pd.isnull(val):
                raise ValueError(f'Missing processing time for ({p}, {e})')
            proc_time[p, e] = float(val)
avail_time = {}
equip_full_cost = {}
for e in equipments:
    row = df[df['Equipment / Cost'] == e].iloc[0]
    atime = row['Available Equipment Operating Time']
    ecost = row['Equipment Cost at Full Load (yuan)']
    if pd.isnull(atime) or pd.isnull(ecost):
        raise ValueError(f'Missing available time or cost for equipment {e}')
    avail_time[e] = float(atime)
    equip_full_cost[e] = float(ecost)
raw_mat_row = df[df['Equipment / Cost'].astype(str).str.strip() == 'Raw Material Cost (yuan/unit)']
if raw_mat_row.empty:
    raise KeyError("Row 'Raw Material Cost (yuan/unit)' not found in CSV.")
raw_mat_cost = {}
for p in products:
    val = raw_mat_row.iloc[0][p]
    if pd.isnull(val):
        raise ValueError(f'Missing raw material cost for {p}')
    raw_mat_cost[p] = float(val)
unit_price_row = df[df['Equipment / Cost'].astype(str).str.strip() == 'Unit Price (yuan/unit)']
if unit_price_row.empty:
    raise KeyError("Row 'Unit Price (yuan/unit)' not found in CSV.")
unit_price = {}
for p in products:
    val = unit_price_row.iloc[0][p]
    if pd.isnull(val):
        raise ValueError(f'Missing unit price for {p}')
    unit_price[p] = float(val)
m = gp.Model('FactoryProductionPlan')
x = m.addVars(feasible_pairs, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
revenue = gp.quicksum((unit_price[p] * x[p, e] for (p, e) in feasible_pairs if equip_proc[e] == 'B'))
raw_mat_total = gp.quicksum((raw_mat_cost[p] * x[p, e] for (p, e) in feasible_pairs if equip_proc[e] == 'B'))
equip_cost = gp.quicksum((equip_full_cost[e] * (gp.quicksum((proc_time[p, e] * x[p, e] for (p2, e2) in feasible_pairs if e2 == e and p2 in products)) / avail_time[e]) for e in equipments))
m.setObjective(revenue - raw_mat_total - equip_cost, gp.GRB.MAXIMIZE)
for e in equipments:
    m.addConstr(gp.quicksum((proc_time[p, e] * x[p, e] for (p2, e2) in feasible_pairs if e2 == e and p2 in products)) <= avail_time[e], name=f'time_{e}')
for p in products:
    sum_A = gp.quicksum((x[p, e] for (pp, e) in feasible_pairs if pp == p and equip_proc[e] == 'A'))
    sum_B = gp.quicksum((x[p, e] for (pp, e) in feasible_pairs if pp == p and equip_proc[e] == 'B'))
    m.addConstr(sum_A == sum_B, name=f'flow_{p}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total profit: {m.objVal:.2f} yuan')
    print('\n--- Production Plan (units processed on each equipment) ---')
    for (p, e) in feasible_pairs:
        val = x[p, e].X
        if val > 1e-06:
            print(f'  {p} on {e}: {val:.2f} units')
    print('\n--- Finished Product Quantities ---')
    for p in products:
        finished = sum((x[p, e].X for (pp, e) in feasible_pairs if pp == p and equip_proc[e] == 'B'))
        print(f'  {p}: {finished:.2f} units')
    print('\n--- Equipment Utilization ---')
    for e in equipments:
        used_time = sum((proc_time[p, e] * x[p, e].X for (p2, e2) in feasible_pairs if e2 == e and p2 in products))
        util = used_time / avail_time[e]
        print(f'  {e}: Used {used_time:.2f} / {avail_time[e]:.2f} hours ({util * 100:.1f}%)')
    print('\n--- Equipment Cost Breakdown ---')
    for e in equipments:
        used_time = sum((proc_time[p, e] * x[p, e].X for (p2, e2) in feasible_pairs if e2 == e and p2 in products))
        cost = equip_full_cost[e] * (used_time / avail_time[e])
        print(f'  {e}: {cost:.2f} yuan')
    print('\n--- Raw Material Cost Breakdown ---')
    for p in products:
        finished = sum((x[p, e].X for (pp, e) in feasible_pairs if pp == p and equip_proc[e] == 'B'))
        cost = raw_mat_cost[p] * finished
        print(f'  {p}: {cost:.2f} yuan')
    print('\n--- Revenue Breakdown ---')
    for p in products:
        finished = sum((x[p, e].X for (pp, e) in feasible_pairs if pp == p and equip_proc[e] == 'B'))
        rev = unit_price[p] * finished
        print(f'  {p}: {rev:.2f} yuan')
else:
    print(f'No optimal solution found. Status: {m.status}')