import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv'
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
products = ['Product I', 'Product II', 'Product III']
product_short = {'Product I': 'I', 'Product II': 'II', 'Product III': 'III'}
procedures = ['A', 'B']
equipment_rows = df['Equipment / Cost'].str.strip()
A_equip = [row for row in equipment_rows if re.fullmatch('A\\d+', row.strip(), re.IGNORECASE)]
B_equip = [row for row in equipment_rows if re.fullmatch('B\\d+', row.strip(), re.IGNORECASE)]
equipments = A_equip + B_equip
equip_to_proc = {}
for e in A_equip:
    equip_to_proc[e] = 'A'
for e in B_equip:
    equip_to_proc[e] = 'B'
proc_time = {}
for e in equipments:
    row = df[df['Equipment / Cost'].str.strip() == e].iloc[0]
    for p in products:
        val = row[p].strip()
        if val != '':
            proc_time[p, e] = float(val)
avail_time = {}
equip_cost_full = {}
for e in equipments:
    row = df[df['Equipment / Cost'].str.strip() == e].iloc[0]
    avail_time[e] = float(row['Available Equipment Operating Time'].strip())
    equip_cost_full[e] = float(row['Equipment Cost at Full Load (yuan)'].strip())

def find_row_idx(label):
    idx = df['Equipment / Cost'].str.strip().str.casefold() == label.strip().casefold()
    if not idx.any():
        raise ValueError(f"Row '{label}' not found in CSV.")
    return df[idx].iloc[0]
raw_mat_row = find_row_idx('Raw Material Cost (yuan/unit)')
unit_price_row = find_row_idx('Unit Price (yuan/unit)')
raw_mat_cost = {}
unit_price = {}
for p in products:
    val_rm = raw_mat_row[p].strip()
    val_up = unit_price_row[p].strip()
    if val_rm == '' or val_up == '':
        raise ValueError(f'Missing raw material cost or unit price for {p}')
    raw_mat_cost[p] = float(val_rm)
    unit_price[p] = float(val_up)
elig_equip = {}
elig_equip['Product I', 'A'] = ['A1', 'A2']
elig_equip['Product II', 'A'] = ['A1', 'A2']
elig_equip['Product III', 'A'] = ['A2']
elig_equip['Product I', 'B'] = ['B1', 'B2', 'B3']
elig_equip['Product II', 'B'] = ['B1']
elig_equip['Product III', 'B'] = ['B2']
x_vars = {}
for p in products:
    for proc in procedures:
        for e in elig_equip[p, proc]:
            if (p, e) in proc_time:
                x_vars[p, e] = None
m = gp.Model('ProductionPlan')
for key in x_vars:
    (p, e) = key
    x_vars[key] = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name=f'x_{product_short[p]}_{e}')
m.update()
for p in products:
    sum_A = gp.quicksum((x_vars[p, e] for e in elig_equip[p, 'A'] if (p, e) in x_vars))
    sum_B = gp.quicksum((x_vars[p, e] for e in elig_equip[p, 'B'] if (p, e) in x_vars))
    m.addConstr(sum_A == sum_B, name=f'flow_{product_short[p]}')
for e in equipments:
    total_time = gp.quicksum((proc_time[p, e] * x_vars[p, e] for p in products if (p, e) in x_vars))
    m.addConstr(total_time <= avail_time[e], name=f'equip_time_{e}')
prod_total = {}
for p in products:
    prod_total[p] = gp.quicksum((x_vars[p, e] for e in elig_equip[p, 'A'] if (p, e) in x_vars))
revenue = gp.quicksum((unit_price[p] * prod_total[p] for p in products))
raw_cost = gp.quicksum((raw_mat_cost[p] * prod_total[p] for p in products))
equip_cost = gp.LinExpr()
for e in equipments:
    total_time = gp.quicksum((proc_time[p, e] * x_vars[p, e] for p in products if (p, e) in x_vars))
    equip_cost += equip_cost_full[e] * (total_time / avail_time[e])
m.setObjective(revenue - raw_cost - equip_cost, gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total profit: {m.objVal:.2f}\n')
    print('--- Production Plan ---')
    for p in products:
        total = prod_total[p].getValue()
        print(f'Product {product_short[p]}: {total:.2f} units produced')
        for proc in procedures:
            for e in elig_equip[p, proc]:
                if (p, e) in x_vars and x_vars[p, e].X > 1e-06:
                    print(f'  {proc} on {e}: {x_vars[p, e].X:.2f} units')
    print('\n--- Equipment Usage ---')
    for e in equipments:
        used_time = sum((proc_time[p, e] * x_vars[p, e].X for p in products if (p, e) in x_vars))
        print(f'{e}: Used {used_time:.2f} / {avail_time[e]:.2f} hours, Cost: {equip_cost_full[e] * (used_time / avail_time[e]):.2f} yuan')
else:
    print(f'No optimal solution found. Status: {m.status}')