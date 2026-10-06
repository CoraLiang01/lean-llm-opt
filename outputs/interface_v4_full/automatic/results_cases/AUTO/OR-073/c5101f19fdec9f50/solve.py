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
equip_proc = {}
for e in equip_A:
    equip_proc[e] = 'A'
for e in equip_B:
    equip_proc[e] = 'B'

def get_row_idx(label):
    idx = df['Equipment / Cost'].astype(str).str.strip().str.casefold() == label.casefold()
    matches = np.where(idx)[0]
    if len(matches) == 0:
        raise KeyError(f"Row '{label}' not found in CSV.")
    return matches[0]
proc_time = {}
for e in equip_all:
    row_idx = get_row_idx(e)
    for p in products:
        val = df.at[row_idx, p]
        if not (pd.isna(val) or str(val).strip() == ''):
            proc_time[product_short[p], e] = float(val)
avail_time = {}
for e in equip_all:
    row_idx = get_row_idx(e)
    val = df.at[row_idx, 'Available Equipment Operating Time']
    if not (pd.isna(val) or str(val).strip() == ''):
        avail_time[e] = float(val)
equip_cost = {}
for e in equip_all:
    row_idx = get_row_idx(e)
    val = df.at[row_idx, 'Equipment Cost at Full Load (yuan)']
    if not (pd.isna(val) or str(val).strip() == ''):
        equip_cost[e] = float(val)
row_idx_rm = get_row_idx('Raw Material Cost (yuan/unit)')
raw_mat_cost = {}
for p in products:
    val = df.at[row_idx_rm, p]
    if not (pd.isna(val) or str(val).strip() == ''):
        raw_mat_cost[product_short[p]] = float(val)
    else:
        raise KeyError(f'Missing raw material cost for {p}')
row_idx_up = get_row_idx('Unit Price (yuan/unit)')
unit_price = {}
for p in products:
    val = df.at[row_idx_up, p]
    if not (pd.isna(val) or str(val).strip() == ''):
        unit_price[product_short[p]] = float(val)
    else:
        raise KeyError(f'Missing unit price for {p}')
compat = {('I', 'A'): ['A1', 'A2'], ('I', 'B'): ['B1', 'B2', 'B3'], ('II', 'A'): ['A1', 'A2'], ('II', 'B'): ['B1'], ('III', 'A'): ['A2'], ('III', 'B'): ['B2']}
var_tuples = []
for p in ['I', 'II', 'III']:
    for proc in procedures:
        for e in compat.get((p, proc), []):
            var_tuples.append((p, proc, e))
m = gp.Model('FactoryProductionPlan')
x = m.addVars(var_tuples, lb=0.0, name='')
for p in ['I', 'II', 'III']:
    sum_A = gp.quicksum((x[p, 'A', e] for e in compat.get((p, 'A'), [])))
    sum_B = gp.quicksum((x[p, 'B', e] for e in compat.get((p, 'B'), [])))
    m.addConstr(sum_A == sum_B, name=f'proc_balance_{p}')
for e in equip_all:
    time_used = gp.LinExpr()
    for p, proc, eq in var_tuples:
        if eq == e:
            time_used += proc_time[p, e] * x[p, proc, e]
    m.addConstr(time_used <= avail_time[e], name=f'equip_time_{e}')
prod_qty = {}
for p in ['I', 'II', 'III']:
    prod_qty[p] = gp.quicksum((x[p, 'A', e] for e in compat.get((p, 'A'), [])))
total_revenue = gp.quicksum((unit_price[p] * prod_qty[p] for p in ['I', 'II', 'III']))
total_rawmat = gp.quicksum((raw_mat_cost[p] * prod_qty[p] for p in ['I', 'II', 'III']))
equip_cost_expr = gp.LinExpr()
for e in equip_all:
    time_used = gp.LinExpr()
    for p, proc, eq in var_tuples:
        if eq == e:
            time_used += proc_time[p, e] * x[p, proc, e]
    if avail_time[e] > 0:
        equip_cost_expr += time_used / avail_time[e] * equip_cost[e]
profit = total_revenue - total_rawmat - equip_cost_expr
m.setObjective(profit, gp.GRB.MAXIMIZE)
m.optimize()