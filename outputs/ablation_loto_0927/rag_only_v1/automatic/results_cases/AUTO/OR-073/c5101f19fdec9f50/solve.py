import pandas as pd
import numpy as np
from gurobipy import Model, GRB
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv'
df = pd.read_csv(csv_path, sep=',')
products = ['Product I', 'Product II', 'Product III']
procedures = ['A', 'B']
equip_A = ['A1', 'A2']
equip_B = ['B1', 'B2', 'B3']
equip_procedure = {}
for e in equip_A:
    equip_procedure[e] = 'A'
for e in equip_B:
    equip_procedure[e] = 'B'
equipment_rows = df[~df['Equipment / Cost'].str.contains('Raw Material Cost|Unit Price', na=False)]
equipment_ids = equipment_rows['Equipment / Cost'].str.strip().tolist()
equip_available_time = {}
equip_full_load_cost = {}
for (idx, row) in equipment_rows.iterrows():
    e = str(row['Equipment / Cost']).strip()
    if e in equip_A or e in equip_B:
        if not pd.isna(row['Available Equipment Operating Time']):
            equip_available_time[e] = float(row['Available Equipment Operating Time'])
        else:
            raise ValueError(f'Missing available time for equipment {e}')
        if not pd.isna(row['Equipment Cost at Full Load (yuan)']):
            equip_full_load_cost[e] = float(row['Equipment Cost at Full Load (yuan)'])
        else:
            raise ValueError(f'Missing full load cost for equipment {e}')
proc_time = {}
for (idx, row) in equipment_rows.iterrows():
    e = str(row['Equipment / Cost']).strip()
    if e in equip_A or e in equip_B:
        for p in products:
            val = row[p]
            if not pd.isna(val):
                proc_time[e, p] = float(val)

def get_param_row(param_name):
    row = df[df['Equipment / Cost'].str.strip() == param_name]
    if row.empty:
        raise ValueError(f"Parameter row '{param_name}' not found in CSV")
    return row.iloc[0]
raw_material_cost = {}
unit_price = {}
rmc_row = get_param_row('Raw Material Cost (yuan/unit)')
up_row = get_param_row('Unit Price (yuan/unit)')
for p in products:
    if not pd.isna(rmc_row[p]):
        raw_material_cost[p] = float(rmc_row[p])
    else:
        raise ValueError(f'Missing raw material cost for {p}')
    if not pd.isna(up_row[p]):
        unit_price[p] = float(up_row[p])
    else:
        raise ValueError(f'Missing unit price for {p}')
eligible_equip = {}
eligible_equip['Product I', 'A'] = ['A1', 'A2']
eligible_equip['Product II', 'A'] = ['A1', 'A2']
eligible_equip['Product III', 'A'] = ['A2']
eligible_equip['Product I', 'B'] = ['B1', 'B2', 'B3']
eligible_equip['Product II', 'B'] = ['B1']
eligible_equip['Product III', 'B'] = ['B2']
eligible_pairs = []
for p in products:
    for s in procedures:
        for e in eligible_equip[p, s]:
            if (e, p) in proc_time:
                eligible_pairs.append((e, p, s))
m = Model('factory_production')
m.Params.OutputFlag = 0
x = m.addVars(products, lb=0, vtype=GRB.CONTINUOUS, name='')
y = {}
for (e, p, s) in eligible_pairs:
    y[e, p, s] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f'y_{e}_{p}_{s}')
m.update()
for p in products:
    for s in procedures:
        eq_equip = eligible_equip[p, s]
        eq_equip_valid = [e for e in eq_equip if (e, p) in proc_time]
        m.addConstr(sum((y[e, p, s] for e in eq_equip_valid)) == x[p], name=f'proc_consistency_{p}_{s}')
for e in equip_A + equip_B:
    relevant_pairs = [(p, s) for p in products for s in procedures if (e, p, s) in y]
    m.addConstr(sum((proc_time[e, p] * y[e, p, s] for (p, s) in relevant_pairs)) <= equip_available_time[e], name=f'equip_capacity_{e}')
profit_expr = sum(((unit_price[p] - raw_material_cost[p]) * x[p] for p in products))
equip_cost_expr = 0
for e in equip_A + equip_B:
    relevant_pairs = [(p, s) for p in products for s in procedures if (e, p, s) in y]
    total_time_used = sum((proc_time[e, p] * y[e, p, s] for (p, s) in relevant_pairs))
    equip_cost_expr += equip_full_load_cost[e] * (total_time_used / equip_available_time[e])
m.setObjective(profit_expr - equip_cost_expr, GRB.MAXIMIZE)
m.optimize()