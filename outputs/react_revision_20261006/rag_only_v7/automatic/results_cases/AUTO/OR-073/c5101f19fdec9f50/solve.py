import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv'
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
products = ['Product I', 'Product II', 'Product III']
product_short = {'Product I': 'I', 'Product II': 'II', 'Product III': 'III'}
procedures = ['A', 'B']
equip_A = ['A1', 'A2']
equip_B = ['B1', 'B2', 'B3']
equipment_procedure = {}
for e in equip_A:
    equipment_procedure[e] = 'A'
for e in equip_B:
    equipment_procedure[e] = 'B'
allowed_equipment = {('I', 'A'): ['A1', 'A2'], ('I', 'B'): ['B1', 'B2', 'B3'], ('II', 'A'): ['A1', 'A2'], ('II', 'B'): ['B1'], ('III', 'A'): ['A2'], ('III', 'B'): ['B2']}
df['Equipment / Cost_cf'] = df['Equipment / Cost'].str.casefold().str.strip()
equipment_rows = df[~df['Available Equipment Operating Time'].eq('') & ~df['Equipment / Cost_cf'].str.contains('raw material') & ~df['Equipment / Cost_cf'].str.contains('unit price')]
equipment_ids = equipment_rows['Equipment / Cost'].tolist()
proc_time = {}
for (_, row) in equipment_rows.iterrows():
    e = row['Equipment / Cost'].strip()
    for p in products:
        p_short = product_short[p]
        val = row[p].strip()
        if val != '':
            proc_time[p_short, e] = float(val)
avail_time = {}
equip_cost_full = {}
for (_, row) in equipment_rows.iterrows():
    e = row['Equipment / Cost'].strip()
    avail_time[e] = float(row['Available Equipment Operating Time'])
    equip_cost_full[e] = float(row['Equipment Cost at Full Load (yuan)'])

def get_param_row(param_name):
    idx = df['Equipment / Cost_cf'] == param_name.casefold()
    if not idx.any():
        raise ValueError(f"Parameter row '{param_name}' not found in CSV.")
    return df[idx].iloc[0]
raw_material_cost_row = get_param_row('Raw Material Cost (yuan/unit)')
unit_price_row = get_param_row('Unit Price (yuan/unit)')
raw_material_cost = {}
unit_price = {}
for p in products:
    p_short = product_short[p]
    val_rm = raw_material_cost_row[p].strip()
    val_up = unit_price_row[p].strip()
    if val_rm == '' or val_up == '':
        raise ValueError(f'Missing raw material cost or unit price for {p}.')
    raw_material_cost[p_short] = float(val_rm)
    unit_price[p_short] = float(val_up)
product_keys = list(product_short.values())
allowed_equip_A = {p: allowed_equipment[p, 'A'] for p in product_short.values()}
allowed_equip_B = {p: allowed_equipment[p, 'B'] for p in product_short.values()}
x_A_keys = [(p, e) for p in product_short.values() for e in allowed_equip_A[p]]
x_B_keys = [(p, e) for p in product_short.values() for e in allowed_equip_B[p]]
m = gp.Model('factory_mixture3')
m.Params.MIPGap = 0.0001
x_A_vars = m.addVars(x_A_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
x_B_vars = m.addVars(x_B_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
q_vars = m.addVars(product_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
for p in product_keys:
    m.addConstr(gp.quicksum((x_A_vars[p, e] for e in allowed_equip_A[p])) == q_vars[p], name=f'flowA_{p}')
    m.addConstr(gp.quicksum((x_B_vars[p, e] for e in allowed_equip_B[p])) == q_vars[p], name=f'flowB_{p}')
for e in equipment_ids:
    if e in equip_A:
        used_time = gp.quicksum((proc_time[p, e] * x_A_vars[p, e] for p in product_keys if (p, e) in x_A_vars))
    elif e in equip_B:
        used_time = gp.quicksum((proc_time[p, e] * x_B_vars[p, e] for p in product_keys if (p, e) in x_B_vars))
    else:
        continue
    m.addConstr(used_time <= avail_time[e], name=f'time_{e}')
revenue = gp.quicksum((unit_price[p] * q_vars[p] for p in product_keys))
raw_mat_cost = gp.quicksum((raw_material_cost[p] * q_vars[p] for p in product_keys))
equip_oper_cost_terms = []
for e in equipment_ids:
    if e in equip_A:
        usage = gp.quicksum((proc_time[p, e] * x_A_vars[p, e] for p in product_keys if (p, e) in x_A_vars))
    elif e in equip_B:
        usage = gp.quicksum((proc_time[p, e] * x_B_vars[p, e] for p in product_keys if (p, e) in x_B_vars))
    else:
        continue
    equip_oper_cost_terms.append(equip_cost_full[e] * (usage / avail_time[e]))
equip_oper_cost = gp.quicksum(equip_oper_cost_terms)
profit = revenue - raw_mat_cost - equip_oper_cost
m.setObjective(profit, GRB.MAXIMIZE)
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')