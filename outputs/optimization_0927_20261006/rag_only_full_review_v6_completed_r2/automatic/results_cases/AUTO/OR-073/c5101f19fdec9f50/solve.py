import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv'
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
df.columns = [c.strip() for c in df.columns]
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
eligible_equipment = {('Product I', 'A'): ['A1', 'A2'], ('Product I', 'B'): ['B1', 'B2', 'B3'], ('Product II', 'A'): ['A1', 'A2'], ('Product II', 'B'): ['B1'], ('Product III', 'A'): ['A2'], ('Product III', 'B'): ['B2']}
equipment_rows = df[df['Equipment / Cost'].str.strip().isin(equip_A + equip_B)].copy()
equipment_rows['Equipment / Cost'] = equipment_rows['Equipment / Cost'].str.strip()
proc_time_per_unit = {}
for (_, row) in equipment_rows.iterrows():
    eq = row['Equipment / Cost']
    for p in products:
        val = row[p].strip()
        if val != '':
            proc_time_per_unit[p, eq] = float(val)
avail_time = {}
for (_, row) in equipment_rows.iterrows():
    eq = row['Equipment / Cost']
    val = row['Available Equipment Operating Time'].strip()
    if val != '':
        avail_time[eq] = float(val)
equip_cost_full_load = {}
for (_, row) in equipment_rows.iterrows():
    eq = row['Equipment / Cost']
    val = row['Equipment Cost at Full Load (yuan)'].strip()
    if val != '':
        equip_cost_full_load[eq] = float(val)
raw_mat_row = df[df['Equipment / Cost'].str.strip().str.casefold() == 'raw material cost (yuan/unit)']
unit_price_row = df[df['Equipment / Cost'].str.strip().str.casefold() == 'unit price (yuan/unit)']
if raw_mat_row.empty or unit_price_row.empty:
    raise ValueError('Missing raw material cost or unit price row in CSV.')
raw_mat_cost = {}
unit_price = {}
for p in products:
    val_rm = raw_mat_row.iloc[0][p].strip()
    val_up = unit_price_row.iloc[0][p].strip()
    if val_rm == '' or val_up == '':
        raise ValueError(f'Missing raw material cost or unit price for {p}')
    raw_mat_cost[p] = float(val_rm)
    unit_price[p] = float(val_up)
for (p, proc) in eligible_equipment:
    for eq in eligible_equipment[p, proc]:
        if (p, eq) not in proc_time_per_unit:
            raise ValueError(f'Missing processing time per unit for {p} on {eq}')
        if eq not in avail_time:
            raise ValueError(f'Missing available operating time for equipment {eq}')
        if eq not in equip_cost_full_load:
            raise ValueError(f'Missing equipment cost at full load for equipment {eq}')
eligible_tuples = []
for ((p, proc), eqs) in eligible_equipment.items():
    for eq in eqs:
        eligible_tuples.append((p, proc, eq))
equipment_to_prodproc = {}
for ((p, proc), eqs) in eligible_equipment.items():
    for eq in eqs:
        equipment_to_prodproc.setdefault(eq, []).append((p, proc))
prodproc_to_equipment = {}
for ((p, proc), eqs) in eligible_equipment.items():
    prodproc_to_equipment[p, proc] = eqs
m = Model('factory_mixture3')
m.Params.OutputFlag = 0
q_vars = m.addVars(products, lb=0, vtype=GRB.CONTINUOUS, name='')
proc_time_vars = m.addVars(eligible_tuples, lb=0, vtype=GRB.CONTINUOUS, name='')
for eq in equip_A + equip_B:
    m.addConstr(quicksum((proc_time_vars[p, proc, eq] for (p, proc) in equipment_to_prodproc.get(eq, []))) <= avail_time[eq], name=f'eq_time_{eq}')
for ((p, proc), eqs) in prodproc_to_equipment.items():
    m.addConstr(quicksum((proc_time_vars[p, proc, eq] for eq in eqs)) == quicksum((proc_time_per_unit[p, eq] * q_vars[p] for eq in eqs)), name=f'prod_proc_link_{p}_{proc}')
revenue = quicksum((unit_price[p] * q_vars[p] for p in products))
raw_cost = quicksum((raw_mat_cost[p] * q_vars[p] for p in products))
equip_cost = quicksum((equip_cost_full_load[eq] * (quicksum((proc_time_vars[p, proc, eq] for (p, proc) in equipment_to_prodproc.get(eq, []))) / avail_time[eq]) for eq in equip_A + equip_B))
m.setObjective(revenue - raw_cost - equip_cost, GRB.MAXIMIZE)
m.optimize()