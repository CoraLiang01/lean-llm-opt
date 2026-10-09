import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv'
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
products = ['Product I', 'Product II', 'Product III']
product_short = {'Product I': 'I', 'Product II': 'II', 'Product III': 'III'}
equipment_rows = df['Equipment / Cost'].str.strip().isin(['A1', 'A2', 'B1', 'B2', 'B3'])
equipment_list = df.loc[equipment_rows, 'Equipment / Cost'].str.strip().tolist()
equipment_procedure = {}
for eq in equipment_list:
    if eq.startswith('A'):
        equipment_procedure[eq] = 'A'
    elif eq.startswith('B'):
        equipment_procedure[eq] = 'B'
    else:
        raise ValueError(f'Unknown equipment type: {eq}')
processing_time = {p: {} for p in products}
for eq in equipment_list:
    eq_row = df[df['Equipment / Cost'].str.strip() == eq].iloc[0]
    for p in products:
        val = eq_row[p].strip()
        if val != '':
            try:
                processing_time[p][eq] = float(val)
            except Exception:
                raise ValueError(f'Invalid processing time for {p}, {eq}: {val}')
        else:
            processing_time[p][eq] = None
available_time = {}
equipment_full_cost = {}
for eq in equipment_list:
    eq_row = df[df['Equipment / Cost'].str.strip() == eq].iloc[0]
    atime = eq_row['Available Equipment Operating Time'].strip()
    ecost = eq_row['Equipment Cost at Full Load (yuan)'].strip()
    try:
        available_time[eq] = float(atime)
        equipment_full_cost[eq] = float(ecost)
    except Exception:
        raise ValueError(f'Invalid available time or equipment cost for {eq}: {atime}, {ecost}')

def get_row_value(row_name, product):
    row = df[df['Equipment / Cost'].str.strip() == row_name]
    if row.empty:
        raise ValueError(f"Row '{row_name}' not found in CSV")
    val = row.iloc[0][product].strip()
    if val == '':
        raise ValueError(f'Missing value for {row_name}, {product}')
    return float(val)
raw_material_cost = {p: get_row_value('Raw Material Cost (yuan/unit)', p) for p in products}
selling_price = {p: get_row_value('Unit Price (yuan/unit)', p) for p in products}
eligible_pairs = []
for p in products:
    for eq in equipment_list:
        proc = equipment_procedure[eq]
        eligible = False
        if processing_time[p][eq] is not None:
            if p == 'Product I':
                eligible = True
            elif p == 'Product II':
                if proc == 'A':
                    eligible = True
                elif proc == 'B' and eq == 'B1':
                    eligible = True
            elif p == 'Product III':
                if proc == 'A' and eq == 'A2':
                    eligible = True
                elif proc == 'B' and eq == 'B2':
                    eligible = True
        if eligible:
            eligible_pairs.append((p, eq))
product_equipment_A = {p: [] for p in products}
product_equipment_B = {p: [] for p in products}
for (p, eq) in eligible_pairs:
    proc = equipment_procedure[eq]
    if proc == 'A':
        product_equipment_A[p].append(eq)
    elif proc == 'B':
        product_equipment_B[p].append(eq)
m = gp.Model('ProductionPlan')
x_vars = m.addVars(eligible_pairs, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
for eq in equipment_list:
    relevant_pairs = [(p, eq) for p in products if (p, eq) in x_vars]
    if relevant_pairs:
        m.addConstr(gp.quicksum((processing_time[p][eq] * x_vars[p, eq] for (p, eq) in relevant_pairs)) <= available_time[eq], name=f'EquipTime_{eq}')
for p in products:
    sum_A = gp.quicksum((x_vars[p, eq] for eq in product_equipment_A[p]))
    sum_B = gp.quicksum((x_vars[p, eq] for eq in product_equipment_B[p]))
    m.addConstr(sum_A == sum_B, name=f'ProcSync_{p}')
total_revenue = gp.quicksum((selling_price[p] * gp.quicksum((x_vars[p, eq] for eq in product_equipment_A[p])) for p in products))
total_raw_cost = gp.quicksum((raw_material_cost[p] * gp.quicksum((x_vars[p, eq] for eq in product_equipment_A[p])) for p in products))
equipment_cost_terms = []
for eq in equipment_list:
    relevant_pairs = [(p, eq) for p in products if (p, eq) in x_vars]
    if relevant_pairs:
        usage = gp.quicksum((processing_time[p][eq] * x_vars[p, eq] for (p, eq) in relevant_pairs))
        equipment_cost_terms.append(equipment_full_cost[eq] * (usage / available_time[eq]))
total_equipment_cost = gp.quicksum(equipment_cost_terms)
m.setObjective(total_revenue - total_raw_cost - total_equipment_cost, gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total profit: {m.objVal:.2f}\n')
    print('Production Plan (units produced per product):')
    for p in products:
        produced = sum((x_vars[p, eq].X for eq in product_equipment_A[p]))
        print(f'  {p}: {produced:.2f}')
    print('\nDetailed assignment (units per product/equipment):')
    for (p, eq) in eligible_pairs:
        val = x_vars[p, eq].X
        if val > 1e-06:
            print(f'  {p} on {eq}: {val:.2f}')
    print('\nEquipment usage (hours):')
    for eq in equipment_list:
        relevant_pairs = [(p, eq) for p in products if (p, eq) in x_vars]
        usage = sum((processing_time[p][eq] * x_vars[p, eq].X for (p, eq) in relevant_pairs))
        print(f'  {eq}: {usage:.2f} / {available_time[eq]:.2f} hours')
    print('\nTotal equipment cost: {:.2f}'.format(total_equipment_cost.getValue()))
    print('Total raw material cost: {:.2f}'.format(total_raw_cost.getValue()))
    print('Total revenue: {:.2f}'.format(total_revenue.getValue()))
else:
    print(f'No optimal solution found. Status: {m.status}')