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
equipments = equip_A + equip_B
equip_to_proc = {}
for e in equip_A:
    equip_to_proc[e] = 'A'
for e in equip_B:
    equip_to_proc[e] = 'B'

def find_row_idx(df, col, value):
    idx = df[df[col].astype(str).str.strip().str.casefold() == value.strip().casefold()].index
    if len(idx) == 0:
        raise KeyError(f"Row '{value}' not found in column '{col}'")
    return idx[0]

def is_equipment_row(s):
    return s.strip() in equipments
equipment_rows = df['Equipment / Cost'].astype(str).apply(is_equipment_row)
equip_df = df[equipment_rows].copy()
avail_time = {}
equip_cost_full = {}
for _, row in equip_df.iterrows():
    e = row['Equipment / Cost'].strip()
    avail_time[e] = float(row['Available Equipment Operating Time'])
    equip_cost_full[e] = float(row['Equipment Cost at Full Load (yuan)'])
proc_time = {}
for _, row in equip_df.iterrows():
    e = row['Equipment / Cost'].strip()
    for p in products:
        val = row[p]
        if not (pd.isna(val) or str(val).strip() == ''):
            proc_time[e, p] = float(val)
rmc_idx = find_row_idx(df, 'Equipment / Cost', 'Raw Material Cost (yuan/unit)')
up_idx = find_row_idx(df, 'Equipment / Cost', 'Unit Price (yuan/unit)')
raw_mat_cost = {}
unit_price = {}
for p in products:
    raw_mat_cost[p] = float(df.at[rmc_idx, p])
    unit_price[p] = float(df.at[up_idx, p])
allowed_equipments = {'Product I': {'A': ['A1', 'A2'], 'B': ['B1', 'B2', 'B3']}, 'Product II': {'A': ['A1', 'A2'], 'B': ['B1']}, 'Product III': {'A': ['A2'], 'B': ['B2']}}
feasible_equip_prod = set()
for p in products:
    for proc in procedures:
        for e in allowed_equipments[p][proc]:
            feasible_equip_prod.add((e, p))
m = gp.Model('FactoryProductionPlan')
q = m.addVars(products, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
z = m.addVars([(e, p) for e, p in feasible_equip_prod], lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
for p in products:
    eqs_A = allowed_equipments[p]['A']
    m.addConstr(gp.quicksum((z[e, p] for e in eqs_A)) == q[p], name=f'flow_A_{p}')
    eqs_B = allowed_equipments[p]['B']
    m.addConstr(gp.quicksum((z[e, p] for e in eqs_B)) == q[p], name=f'flow_B_{p}')
for e in equipments:
    prods = [p for p in products if (e, p) in feasible_equip_prod]
    if not prods:
        continue
    m.addConstr(gp.quicksum((proc_time[e, p] * z[e, p] for p in prods)) <= avail_time[e], name=f'time_{e}')
revenue = gp.quicksum((unit_price[p] * q[p] for p in products))
raw_cost = gp.quicksum((raw_mat_cost[p] * q[p] for p in products))
equip_cost = gp.quicksum((gp.quicksum((proc_time[e, p] * z[e, p] for p in products if (e, p) in feasible_equip_prod)) / avail_time[e] * equip_cost_full[e] for e in equipments))
profit = revenue - raw_cost - equip_cost
m.setObjective(profit, gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total profit: {m.objVal:.2f}')
    print('\n--- Production Quantities (q) ---')
    for p in products:
        print(f'  {p}: {q[p].X:.4f}')
    print('\n--- Equipment Assignment (z) ---')
    for e, p in sorted(feasible_equip_prod):
        val = z[e, p].X
        if val > 1e-06:
            print(f'  {e} processes {p}: {val:.4f}')
    print('\n--- Equipment Utilization ---')
    for e in equipments:
        used = sum((proc_time[e, p] * z[e, p].X for p in products if (e, p) in feasible_equip_prod))
        print(f'  {e}: {used:.2f} / {avail_time[e]:.2f} hours used ({100 * used / avail_time[e]:.2f}%)')
else:
    print(f'No optimal solution found. Status: {m.status}')