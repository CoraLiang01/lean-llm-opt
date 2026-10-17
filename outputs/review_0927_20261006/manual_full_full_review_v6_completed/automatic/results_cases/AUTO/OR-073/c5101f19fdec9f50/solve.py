import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv'
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
products = ['Product I', 'Product II', 'Product III']
product_short = {'Product I': 'I', 'Product II': 'II', 'Product III': 'III'}
procedures = ['A', 'B']
equipments_A = ['A1', 'A2']
equipments_B = ['B1', 'B2', 'B3']
eligibility = {'I': {'A': ['A1', 'A2'], 'B': ['B1', 'B2', 'B3']}, 'II': {'A': ['A1', 'A2'], 'B': ['B1']}, 'III': {'A': ['A2'], 'B': ['B2']}}
equipment_rows = df['Equipment / Cost'].str.strip().isin(equipments_A + equipments_B)
equip_df = df[equipment_rows].copy()
equip_df['Equipment / Cost'] = equip_df['Equipment / Cost'].str.strip()
processing_time = {}
for _, row in equip_df.iterrows():
    eq = row['Equipment / Cost']
    for prod in products:
        prod_short = product_short[prod]
        val = row[prod].strip()
        if val != '':
            processing_time[prod_short, eq] = float(val)
available_time = {}
equipment_cost_full = {}
for _, row in equip_df.iterrows():
    eq = row['Equipment / Cost']
    atime = row['Available Equipment Operating Time'].strip()
    ecost = row['Equipment Cost at Full Load (yuan)'].strip()
    if atime != '':
        available_time[eq] = float(atime)
    if ecost != '':
        equipment_cost_full[eq] = float(ecost)

def find_row_idx(df, pattern):
    return df['Equipment / Cost'].str.strip().str.casefold() == pattern.casefold()
raw_mat_row = df[find_row_idx(df, 'Raw Material Cost (yuan/unit)')]
unit_price_row = df[find_row_idx(df, 'Unit Price (yuan/unit)')]
if raw_mat_row.empty or unit_price_row.empty:
    raise ValueError('Raw Material Cost or Unit Price row not found in CSV.')
raw_material_cost = {}
unit_price = {}
for prod in products:
    prod_short = product_short[prod]
    val_rm = raw_mat_row.iloc[0][prod].strip()
    val_up = unit_price_row.iloc[0][prod].strip()
    if val_rm == '' or val_up == '':
        raise ValueError(f'Missing raw material cost or unit price for {prod}.')
    raw_material_cost[prod_short] = float(val_rm)
    unit_price[prod_short] = float(val_up)
x_keys = []
for prod_short in ['I', 'II', 'III']:
    for proc in procedures:
        for eq in eligibility[prod_short][proc]:
            x_keys.append((prod_short, proc, eq))
m = gp.Model('ProductionPlan')
x_vars = m.addVars(x_keys, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
for prod_short in ['I', 'II', 'III']:
    sum_A = gp.quicksum((x_vars[prod_short, 'A', eq] for eq in eligibility[prod_short]['A']))
    sum_B = gp.quicksum((x_vars[prod_short, 'B', eq] for eq in eligibility[prod_short]['B']))
    m.addConstr(sum_A == sum_B, name=f'prod_balance_{prod_short}')
for eq in set(equipments_A + equipments_B):
    relevant_x = []
    for prod_short in ['I', 'II', 'III']:
        for proc in procedures:
            if eq in eligibility[prod_short][proc]:
                relevant_x.append((prod_short, proc, eq))
    if eq not in available_time:
        raise ValueError(f'Missing available time for equipment {eq}.')
    m.addConstr(gp.quicksum((processing_time[prod_short, eq] * x_vars[prod_short, proc, eq] for prod_short, proc, eq in relevant_x)) <= available_time[eq], name=f'eq_time_{eq}')
total_produced = {}
for prod_short in ['I', 'II', 'III']:
    total_produced[prod_short] = gp.quicksum((x_vars[prod_short, 'A', eq] for eq in eligibility[prod_short]['A']))
revenue = gp.quicksum((unit_price[prod_short] * total_produced[prod_short] for prod_short in ['I', 'II', 'III']))
raw_mat_cost = gp.quicksum((raw_material_cost[prod_short] * total_produced[prod_short] for prod_short in ['I', 'II', 'III']))
equipment_cost = []
for eq in set(equipments_A + equipments_B):
    used_time = gp.quicksum((processing_time[prod_short, eq] * x_vars[prod_short, proc, eq] for prod_short in ['I', 'II', 'III'] for proc in procedures if eq in eligibility[prod_short][proc]))
    equipment_cost.append(used_time / available_time[eq] * equipment_cost_full[eq])
total_equipment_cost = gp.quicksum(equipment_cost)
m.setObjective(revenue - raw_mat_cost - total_equipment_cost, gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total profit: {m.objVal:.2f} yuan')
    print('\n--- Production Plan ---')
    for prod_short in ['I', 'II', 'III']:
        prod_name = [k for k, v in product_short.items() if v == prod_short][0]
        qty = total_produced[prod_short].getValue()
        print(f'Product {prod_name}: {qty:.2f} units produced')
        for proc in procedures:
            for eq in eligibility[prod_short][proc]:
                val = x_vars[prod_short, proc, eq].X
                if val > 1e-06:
                    print(f'  Procedure {proc} on {eq}: {val:.2f} units')
    print('\n--- Equipment Utilization ---')
    for eq in set(equipments_A + equipments_B):
        used_time = sum((processing_time[prod_short, eq] * x_vars[prod_short, proc, eq].X for prod_short in ['I', 'II', 'III'] for proc in procedures if eq in eligibility[prod_short][proc]))
        print(f'Equipment {eq}: Used {used_time:.2f} / {available_time[eq]:.2f} hours')
else:
    print(f'No optimal solution found. Status: {m.status}')