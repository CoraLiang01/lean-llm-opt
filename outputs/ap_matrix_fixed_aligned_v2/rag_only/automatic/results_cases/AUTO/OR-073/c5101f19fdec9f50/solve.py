import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv'
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
products = ['Product I', 'Product II', 'Product III']
product_short = {'Product I': 'I', 'Product II': 'II', 'Product III': 'III'}
procedure_A_equip = ['A1', 'A2']
procedure_B_equip = ['B1', 'B2', 'B3']
eligibility = {'I': {'A': ['A1', 'A2'], 'B': ['B1', 'B2', 'B3']}, 'II': {'A': ['A1', 'A2'], 'B': ['B1']}, 'III': {'A': ['A2'], 'B': ['B2']}}
equipment_rows = df[~df['Equipment / Cost'].str.casefold().str.contains('raw material cost|unit price', na=False)]
equipments = equipment_rows['Equipment / Cost'].tolist()
available_time = {}
equip_cost_full_load = {}
for idx, row in equipment_rows.iterrows():
    equip = row['Equipment / Cost']
    atime = row['Available Equipment Operating Time']
    ecost = row['Equipment Cost at Full Load (yuan)']
    if atime != '':
        available_time[equip] = float(atime)
    if ecost != '':
        equip_cost_full_load[equip] = float(ecost)
processing_time = {}
for equip in equipments:
    processing_time[equip] = {}
    row = equipment_rows[equipment_rows['Equipment / Cost'] == equip].iloc[0]
    for prod in products:
        val = row[prod]
        if val != '':
            processing_time[equip][prod_short[prod]] = float(val)
raw_mat_row = df[df['Equipment / Cost'].str.casefold() == 'raw material cost (yuan/unit)']
unit_price_row = df[df['Equipment / Cost'].str.casefold() == 'unit price (yuan/unit)']
if raw_mat_row.empty or unit_price_row.empty:
    raise ValueError('Missing raw material cost or unit price row in CSV.')
raw_material_cost = {}
unit_price = {}
for prod in products:
    prod_short_name = product_short[prod]
    val = raw_mat_row.iloc[0][prod]
    if val == '':
        raise ValueError(f'Missing raw material cost for {prod}')
    raw_material_cost[prod_short_name] = float(val)
    val = unit_price_row.iloc[0][prod]
    if val == '':
        raise ValueError(f'Missing unit price for {prod}')
    unit_price[prod_short_name] = float(val)
z_keys = []
for p_short in ['I', 'II', 'III']:
    for proc, equip_list in eligibility[p_short].items():
        for e in equip_list:
            z_keys.append((p_short, proc, e))
equip_proc_prod = {}
for e in equipments:
    equip_proc_prod[e] = []
    for p_short in ['I', 'II', 'III']:
        for proc in ['A', 'B']:
            if e in eligibility[p_short].get(proc, []):
                equip_proc_prod[e].append((p_short, proc))
m = Model('factory_mixture3')
m.Params.OutputFlag = 0
x_vars = m.addVars(['I', 'II', 'III'], lb=0, vtype=GRB.CONTINUOUS, name='')
z_vars = m.addVars(z_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
for p_short in ['I', 'II', 'III']:
    for proc in ['A', 'B']:
        eligible_equip = eligibility[p_short].get(proc, [])
        if not eligible_equip:
            continue
        m.addConstr(quicksum((z_vars[p_short, proc, e] for e in eligible_equip)) == x_vars[p_short], name=f'prod_proc_{p_short}_{proc}')
for e in equipments:
    if e not in available_time:
        continue
    terms = []
    for p_short, proc in equip_proc_prod[e]:
        if p_short in processing_time.get(e, {}):
            terms.append(processing_time[e][p_short] * z_vars[p_short, proc, e])
    if terms:
        m.addConstr(quicksum(terms) <= available_time[e], name=f'equip_time_{e}')
revenue = quicksum((unit_price[p] * x_vars[p] for p in ['I', 'II', 'III']))
raw_mat_cost = quicksum((raw_material_cost[p] * x_vars[p] for p in ['I', 'II', 'III']))
equip_cost_terms = []
for e in equipments:
    if e not in available_time or e not in equip_cost_full_load:
        continue
    time_used = quicksum((processing_time[e][p_short] * z_vars[p_short, proc, e] for p_short, proc in equip_proc_prod[e] if p_short in processing_time.get(e, {})))
    equip_cost_terms.append(equip_cost_full_load[e] * (time_used / available_time[e]))
equip_cost = quicksum(equip_cost_terms)
profit = revenue - raw_mat_cost - equip_cost
m.setObjective(profit, GRB.MAXIMIZE)
m.optimize()