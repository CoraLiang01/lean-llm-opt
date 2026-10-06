import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv'
df = pd.read_csv(csv_path, sep=',')
products = ['Product I', 'Product II', 'Product III']
product_short = {'Product I': 'I', 'Product II': 'II', 'Product III': 'III'}
procedures = ['A', 'B']
equipment_rows = df['Equipment / Cost'].str.strip()
equipments_A = ['A1', 'A2']
equipments_B = ['B1', 'B2', 'B3']
equipments = equipments_A + equipments_B
equipment_procedure = {}
for e in equipments_A:
    equipment_procedure[e] = 'A'
for e in equipments_B:
    equipment_procedure[e] = 'B'
eligible_equipment = {('Product I', 'A'): ['A1', 'A2'], ('Product I', 'B'): ['B1', 'B2', 'B3'], ('Product II', 'A'): ['A1', 'A2'], ('Product II', 'B'): ['B1'], ('Product III', 'A'): ['A2'], ('Product III', 'B'): ['B2']}

def get_equipment_row(equip_name):
    idx = equipment_rows.str.casefold() == equip_name.casefold()
    matches = np.where(idx)[0]
    if len(matches) == 0:
        raise ValueError(f"Equipment '{equip_name}' not found in CSV.")
    return matches[0]
proc_time = {}
for e in equipments:
    row_idx = get_equipment_row(e)
    for p in products:
        val = df.at[row_idx, p]
        if not np.isnan(val):
            proc_time[e, p] = float(val)
avail_time = {}
for e in equipments:
    row_idx = get_equipment_row(e)
    val = df.at[row_idx, 'Available Equipment Operating Time']
    if np.isnan(val):
        raise ValueError(f"Missing available operating time for equipment '{e}'.")
    avail_time[e] = float(val)
equip_cost_full = {}
for e in equipments:
    row_idx = get_equipment_row(e)
    val = df.at[row_idx, 'Equipment Cost at Full Load (yuan)']
    if np.isnan(val):
        raise ValueError(f"Missing equipment cost at full load for equipment '{e}'.")
    equip_cost_full[e] = float(val)
rmc_row = equipment_rows.str.strip().str.casefold() == 'raw material cost (yuan/unit)'.casefold()
rmc_idx = np.where(rmc_row)[0]
if len(rmc_idx) != 1:
    raise ValueError('Raw Material Cost row not found or not unique.')
raw_material_cost = {}
for p in products:
    val = df.at[rmc_idx[0], p]
    if np.isnan(val):
        raise ValueError(f"Missing raw material cost for product '{p}'.")
    raw_material_cost[p] = float(val)
up_row = equipment_rows.str.strip().str.casefold() == 'unit price (yuan/unit)'.casefold()
up_idx = np.where(up_row)[0]
if len(up_idx) != 1:
    raise ValueError('Unit Price row not found or not unique.')
unit_price = {}
for p in products:
    val = df.at[up_idx[0], p]
    if np.isnan(val):
        raise ValueError(f"Missing unit price for product '{p}'.")
    unit_price[p] = float(val)
z_keys = []
for p in products:
    for q in procedures:
        for e in eligible_equipment[p, q]:
            z_keys.append((e, p))
z_keys = list(dict.fromkeys(z_keys))
m = gp.Model('FactoryProduction')
m.Params.MIPGap = 0.0001
x = m.addVars(products, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
z = m.addVars(z_keys, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
for p in products:
    eqs_A = eligible_equipment[p, 'A']
    m.addConstr(gp.quicksum((z[e, p] for e in eqs_A)) == x[p], name=f'consist_{p}_A')
    eqs_B = eligible_equipment[p, 'B']
    m.addConstr(gp.quicksum((z[e, p] for e in eqs_B)) == x[p], name=f'consist_{p}_B')
for e in equipments:
    eligible_ps = [p for (ee, p) in z_keys if ee == e]
    m.addConstr(gp.quicksum((proc_time[e, p] * z[e, p] for p in eligible_ps)) <= avail_time[e], name=f'timelimit_{e}')
revenue = gp.quicksum((unit_price[p] * x[p] for p in products))
rm_cost = gp.quicksum((raw_material_cost[p] * x[p] for p in products))
equip_cost = gp.quicksum((gp.quicksum((proc_time[e, p] * z[e, p] for p in [p for (ee, p) in z_keys if ee == e])) / avail_time[e] * equip_cost_full[e] for e in equipments))
profit = revenue - rm_cost - equip_cost
m.setObjective(profit, gp.GRB.MAXIMIZE)
m.optimize()
if m.Status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal:.6f}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X:.6f}')
else:
    print(f'Solver status: {m.Status}')