import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv'
df = pd.read_csv(csv_path, sep=',')
products = ['I', 'II', 'III']
product_col_map = {'I': 'Product I', 'II': 'Product II', 'III': 'Product III'}
procedures = ['A', 'B']
equip_A = ['A1', 'A2']
equip_B = ['B1', 'B2', 'B3']
equipment_rows = df['Equipment / Cost'].str.strip()
equipment_set = set(equipment_rows[:7])
compat = {('I', 'A'): ['A1', 'A2'], ('I', 'B'): ['B1', 'B2', 'B3'], ('II', 'A'): ['A1', 'A2'], ('II', 'B'): ['B1'], ('III', 'A'): ['A2'], ('III', 'B'): ['B2']}
processing_time = {}
for eq in equipment_set:
    row = df[df['Equipment / Cost'].str.strip() == eq]
    if row.empty:
        continue
    processing_time[eq] = {}
    for p in products:
        col = product_col_map[p]
        val = row.iloc[0][col]
        if pd.notnull(val):
            processing_time[eq][p] = float(val)
avail_time = {}
equip_cost_full = {}
for eq in equipment_set:
    row = df[df['Equipment / Cost'].str.strip() == eq]
    if row.empty:
        continue
    atime = row.iloc[0]['Available Equipment Operating Time']
    ecost = row.iloc[0]['Equipment Cost at Full Load (yuan)']
    if pd.notnull(atime):
        avail_time[eq] = float(atime)
    if pd.notnull(ecost):
        equip_cost_full[eq] = float(ecost)

def get_row_value(row_name, col):
    row = df[df['Equipment / Cost'].str.strip() == row_name]
    if row.empty:
        raise ValueError(f"Row '{row_name}' not found in CSV.")
    val = row.iloc[0][col]
    if pd.isnull(val):
        raise ValueError(f'Missing value for {row_name}, {col}')
    return float(val)
raw_material_cost = {}
unit_price = {}
for p in products:
    col = product_col_map[p]
    raw_material_cost[p] = get_row_value('Raw Material Cost (yuan/unit)', col)
    unit_price[p] = get_row_value('Unit Price (yuan/unit)', col)
var_tuples = []
for p in products:
    for proc in procedures:
        for eq in compat[p, proc]:
            var_tuples.append((p, proc, eq))
m = gp.Model('FactoryProductionPlan')
x = m.addVars(var_tuples, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
for p in products:
    sum_A = gp.quicksum((x[p, 'A', eq] for eq in compat[p, 'A']))
    sum_B = gp.quicksum((x[p, 'B', eq] for eq in compat[p, 'B']))
    m.addConstr(sum_A == sum_B, name=f'sync_{p}')
for eq in equipment_set:
    relevant_vars = []
    for p in products:
        for proc in procedures:
            if eq in compat.get((p, proc), []):
                relevant_vars.append((p, proc, eq))
    if not relevant_vars:
        continue
    m.addConstr(gp.quicksum((processing_time[eq][p] * x[p, proc, eq] for p, proc, eq in relevant_vars)) <= avail_time[eq], name=f'equip_time_{eq}')
prod_qty = {}
for p in products:
    prod_qty[p] = gp.quicksum((x[p, 'A', eq] for eq in compat[p, 'A']))
total_revenue = gp.quicksum((unit_price[p] * prod_qty[p] for p in products))
total_raw_cost = gp.quicksum((raw_material_cost[p] * prod_qty[p] for p in products))
equip_oper_cost = []
for eq in equipment_set:
    assigned_time = gp.quicksum((processing_time[eq][p] * x[p, proc, eq] for p in products for proc in procedures if (p, proc, eq) in x))
    if eq in avail_time and eq in equip_cost_full:
        equip_oper_cost.append(assigned_time / avail_time[eq] * equip_cost_full[eq])
total_equip_cost = gp.quicksum(equip_oper_cost)
m.setObjective(total_revenue - total_raw_cost - total_equip_cost, gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total profit: {m.objVal:.2f} yuan')
    print('\n--- Production Plan ---')
    for p in products:
        qty = prod_qty[p].getValue()
        print(f'Product {p}: {qty:.2f} units produced')
        for proc in procedures:
            for eq in compat[p, proc]:
                v = x[p, proc, eq].X
                if v > 1e-06:
                    print(f'  {proc} on {eq}: {v:.2f} units')
    print('\n--- Equipment Utilization ---')
    for eq in equipment_set:
        assigned_time = 0.0
        for p in products:
            for proc in procedures:
                if (p, proc, eq) in x:
                    assigned_time += processing_time[eq][p] * x[p, proc, eq].X
        if assigned_time > 1e-06:
            print(f'{eq}: Used {assigned_time:.2f} / {avail_time[eq]:.2f} hours, Cost: {equip_cost_full[eq]:.2f} yuan at full load')
else:
    print(f'No optimal solution found. Status: {m.status}')