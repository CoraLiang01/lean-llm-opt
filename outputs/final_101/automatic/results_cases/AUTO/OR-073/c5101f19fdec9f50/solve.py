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
compat = {('I', 'A'): ['A1', 'A2'], ('II', 'A'): ['A1', 'A2'], ('III', 'A'): ['A2'], ('I', 'B'): ['B1', 'B2', 'B3'], ('II', 'B'): ['B1'], ('III', 'B'): ['B2']}
equipment_rows = df.loc[df['Available Equipment Operating Time'].notnull() & df['Equipment Cost at Full Load (yuan)'].notnull(), 'Equipment / Cost'].astype(str).str.strip().tolist()
equipment_proc = {}
for eq in equipment_rows:
    if eq.startswith('A'):
        equipment_proc[eq] = 'A'
    elif eq.startswith('B'):
        equipment_proc[eq] = 'B'
proc_time = {}
for eq in equipment_rows:
    row = df[df['Equipment / Cost'].astype(str).str.strip() == eq]
    for p in products:
        val = row[p].values[0]
        if not (pd.isnull(val) or str(val).strip() == ''):
            proc_time[product_short[p], eq] = float(val)
avail_time = {}
for eq in equipment_rows:
    row = df[df['Equipment / Cost'].astype(str).str.strip() == eq]
    val = row['Available Equipment Operating Time'].values[0]
    if not (pd.isnull(val) or str(val).strip() == ''):
        avail_time[eq] = float(val)
equip_cost_full = {}
for eq in equipment_rows:
    row = df[df['Equipment / Cost'].astype(str).str.strip() == eq]
    val = row['Equipment Cost at Full Load (yuan)'].values[0]
    if not (pd.isnull(val) or str(val).strip() == ''):
        equip_cost_full[eq] = float(val)
rmc_row = df['Equipment / Cost'].astype(str).str.strip().str.casefold() == 'raw material cost (yuan/unit)'
if not rmc_row.any():
    raise ValueError('Raw Material Cost row not found in CSV.')
raw_mat_cost = {}
for p in products:
    val = df.loc[rmc_row, p].values[0]
    if not (pd.isnull(val) or str(val).strip() == ''):
        raw_mat_cost[product_short[p]] = float(val)
up_row = df['Equipment / Cost'].astype(str).str.strip().str.casefold() == 'unit price (yuan/unit)'
if not up_row.any():
    raise ValueError('Unit Price row not found in CSV.')
unit_price = {}
for p in products:
    val = df.loc[up_row, p].values[0]
    if not (pd.isnull(val) or str(val).strip() == ''):
        unit_price[product_short[p]] = float(val)
var_tuples = []
for p in product_short.values():
    for proc in procedures:
        for eq in compat.get((p, proc), []):
            var_tuples.append((p, proc, eq))
m = gp.Model('FactoryProductionPlan')
x = m.addVars(var_tuples, lb=0.0, name='')
for eq in equipment_rows:
    relevant_vars = []
    for p, proc, eq2 in var_tuples:
        if eq2 == eq:
            relevant_vars.append((p, proc, eq2))
    if not relevant_vars:
        continue
    m.addConstr(gp.quicksum((proc_time[p, eq] * x[p, proc, eq] for p, proc, eq in relevant_vars)) <= avail_time[eq], name=f'time_{eq}')
for p in product_short.values():
    sum_A = gp.quicksum((x[p, 'A', eq] for eq in compat.get((p, 'A'), []) if (p, 'A', eq) in x))
    sum_B = gp.quicksum((x[p, 'B', eq] for eq in compat.get((p, 'B'), []) if (p, 'B', eq) in x))
    m.addConstr(sum_A == sum_B, name=f'flow_{p}')
prod_qty = {}
for p in product_short.values():
    prod_qty[p] = gp.quicksum((x[p, 'A', eq] for eq in compat.get((p, 'A'), []) if (p, 'A', eq) in x))
total_revenue = gp.quicksum((unit_price[p] * prod_qty[p] for p in product_short.values()))
total_rm_cost = gp.quicksum((raw_mat_cost[p] * prod_qty[p] for p in product_short.values()))
equip_oper_cost = []
for eq in equipment_rows:
    used_time = gp.quicksum((proc_time[p, eq] * x[p, proc, eq] for p, proc, eq2 in var_tuples if eq2 == eq))
    equip_oper_cost.append(equip_cost_full[eq] * (used_time / avail_time[eq]))
total_equip_cost = gp.quicksum(equip_oper_cost)
m.setObjective(total_revenue - total_rm_cost - total_equip_cost, gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total profit: {m.objVal:.2f} yuan')
    print('\n--- Production Plan ---')
    for p, proc, eq in var_tuples:
        val = x[p, proc, eq].X
        if val > 1e-06:
            print(f'Product {p}, Procedure {proc}, Equipment {eq}: {val:.2f} units')
    print('\n--- Product Totals ---')
    for p in product_short.values():
        qty = prod_qty[p].getValue()
        print(f'Product {p}: {qty:.2f} units')
    print('\n--- Equipment Utilization ---')
    for eq in equipment_rows:
        used = sum((proc_time[p, eq] * x[p, proc, eq].X for p, proc, eq2 in var_tuples if eq2 == eq))
        print(f'Equipment {eq}: Used {used:.2f} / {avail_time[eq]:.2f} hours, Cost at full load: {equip_cost_full[eq]:.2f} yuan')
else:
    print(f'No optimal solution found. Status: {m.status}')