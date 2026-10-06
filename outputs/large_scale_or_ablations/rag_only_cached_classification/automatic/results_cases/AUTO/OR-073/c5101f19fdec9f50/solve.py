import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
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
equipment_rows = df[df['Equipment / Cost'].isin(equipments)].copy()
equipment_rows['Equipment'] = equipment_rows['Equipment / Cost'].astype(str).str.strip()
raw_mat_row = df['Equipment / Cost'].str.strip().eq('Raw Material Cost (yuan/unit)')
unit_price_row = df['Equipment / Cost'].str.strip().eq('Unit Price (yuan/unit)')
if not raw_mat_row.any() or not unit_price_row.any():
    raise ValueError('Missing required parameter rows in CSV.')
raw_mat_cost = df.loc[raw_mat_row, products].iloc[0].astype(float).to_dict()
unit_price = df.loc[unit_price_row, products].iloc[0].astype(float).to_dict()
proc_time = {}
for _, row in equipment_rows.iterrows():
    e = row['Equipment']
    for p in products:
        val = row[p]
        if not (pd.isna(val) or str(val).strip() == ''):
            proc_time[p, e] = float(val)
avail_time = {}
equip_cost_full = {}
for _, row in equipment_rows.iterrows():
    e = row['Equipment']
    atime = row['Available Equipment Operating Time']
    if pd.isna(atime) or str(atime).strip() == '':
        raise ValueError(f'Missing available time for equipment {e}')
    avail_time[e] = float(atime)
    ecost = row['Equipment Cost at Full Load (yuan)']
    if pd.isna(ecost) or str(ecost).strip() == '':
        raise ValueError(f'Missing equipment cost at full load for equipment {e}')
    equip_cost_full[e] = float(ecost)
eligible = {}
for p in products:
    for e in equipments:
        proc = equip_to_proc[e]
        if p == 'Product I':
            if proc == 'A' and e in ['A1', 'A2'] or (proc == 'B' and e in ['B1', 'B2', 'B3']):
                eligible[p, e] = True
            else:
                eligible[p, e] = False
        elif p == 'Product II':
            if proc == 'A' and e in ['A1', 'A2'] or (proc == 'B' and e == 'B1'):
                eligible[p, e] = True
            else:
                eligible[p, e] = False
        elif p == 'Product III':
            if proc == 'A' and e == 'A2' or (proc == 'B' and e == 'B2'):
                eligible[p, e] = True
            else:
                eligible[p, e] = False
m = Model('factory_production')
x = m.addVars(products, lb=0, vtype=GRB.CONTINUOUS, name='')
y = {}
for p in products:
    for e in equipments:
        if eligible[p, e]:
            y[p, e] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f'y_{product_short[p]}_{e}')
m.update()
for p in products:
    y_sum_A = quicksum((y[p, e] for e in equip_A if eligible[p, e]))
    m.addConstr(y_sum_A == x[p], name=f'assignA_{product_short[p]}')
    y_sum_B = quicksum((y[p, e] for e in equip_B if eligible[p, e]))
    m.addConstr(y_sum_B == x[p], name=f'assignB_{product_short[p]}')
for e in equipments:
    y_sum = quicksum((proc_time[p, e] * y[p, e] for p in products if eligible[p, e] and (p, e) in proc_time))
    m.addConstr(y_sum <= avail_time[e], name=f'time_{e}')
revenue = quicksum((unit_price[p] * x[p] for p in products))
rawmat = quicksum((raw_mat_cost[p] * x[p] for p in products))
used_time = {}
for e in equipments:
    used_time[e] = quicksum((proc_time[p, e] * y[p, e] for p in products if eligible[p, e] and (p, e) in proc_time))
equip_cost = quicksum((used_time[e] / avail_time[e] * equip_cost_full[e] for e in equipments))
m.setObjective(revenue - rawmat - equip_cost, GRB.MAXIMIZE)
m.optimize()