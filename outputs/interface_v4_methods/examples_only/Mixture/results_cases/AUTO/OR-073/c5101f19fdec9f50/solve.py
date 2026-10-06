import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv'
df = pd.read_csv(csv_path, sep=',')
products = ['Product I', 'Product II', 'Product III']
product_short = {'Product I': 'I', 'Product II': 'II', 'Product III': 'III'}
equip_A = ['A1', 'A2']
equip_B = ['B1', 'B2', 'B3']
equipments = equip_A + equip_B
equip_proc = {}
for e in equip_A:
    equip_proc[e] = 'A'
for e in equip_B:
    equip_proc[e] = 'B'

def row_idx(label):
    idx = df['Equipment / Cost'].astype(str).str.strip().str.casefold() == label.strip().casefold()
    matches = np.where(idx)[0]
    if len(matches) == 0:
        raise KeyError(f"Row '{label}' not found in 'Equipment / Cost'")
    return matches[0]
proc_time = {}
for e in equipments:
    i = row_idx(e)
    for p in products:
        val = df.at[i, p]
        if not (pd.isna(val) or str(val).strip() == ''):
            proc_time[e, p] = float(val)
avail_time = {}
equip_cost_full = {}
for e in equipments:
    i = row_idx(e)
    avail_time[e] = float(df.at[i, 'Available Equipment Operating Time'])
    equip_cost_full[e] = float(df.at[i, 'Equipment Cost at Full Load (yuan)'])
i_rm = row_idx('Raw Material Cost (yuan/unit)')
raw_mat_cost = {}
for p in products:
    val = df.at[i_rm, p]
    if pd.isna(val) or str(val).strip() == '':
        raise KeyError(f'Missing raw material cost for {p}')
    raw_mat_cost[p] = float(val)
i_price = row_idx('Unit Price (yuan/unit)')
unit_price = {}
for p in products:
    val = df.at[i_price, p]
    if pd.isna(val) or str(val).strip() == '':
        raise KeyError(f'Missing unit price for {p}')
    unit_price[p] = float(val)
eligible_equip = []
for p in products:
    if p == 'Product I':
        eligible_equip += [(e, p) for e in ['A1', 'A2']]
        eligible_equip += [(e, p) for e in ['B1', 'B2', 'B3']]
    elif p == 'Product II':
        eligible_equip += [(e, p) for e in ['A1', 'A2']]
        eligible_equip += [('B1', p)]
    elif p == 'Product III':
        eligible_equip += [('A2', p)]
        eligible_equip += [('B2', p)]
for e, p in eligible_equip:
    if (e, p) not in proc_time:
        raise KeyError(f'Missing processing time for equipment {e}, product {p}')
m = gp.Model('FactoryProductionPlan')
x = m.addVars(products, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
for e in equipments:
    eligible_ps = [p for ee, p in eligible_equip if ee == e]
    if eligible_ps:
        m.addConstr(gp.quicksum((proc_time[e, p] * x[p] for p in eligible_ps)) <= avail_time[e], name=f'time_{e}')
revenue = gp.quicksum((unit_price[p] * x[p] for p in products))
raw_cost = gp.quicksum((raw_mat_cost[p] * x[p] for p in products))
equip_cost = gp.quicksum((gp.quicksum((proc_time[e, p] * x[p] for p in [p for ee, p in eligible_equip if ee == e])) / avail_time[e] * equip_cost_full[e] for e in equipments if any((ee == e for ee, p in eligible_equip))))
profit = revenue - raw_cost - equip_cost
m.setObjective(profit, gp.GRB.MAXIMIZE)
m.optimize()