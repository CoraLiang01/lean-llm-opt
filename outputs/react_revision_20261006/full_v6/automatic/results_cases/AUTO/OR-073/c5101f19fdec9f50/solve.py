import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv'
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
products = ['Product I', 'Product II', 'Product III']
product_short = {'Product I': 'I', 'Product II': 'II', 'Product III': 'III'}
procedures = ['A', 'B']
A_equipment = ['A1', 'A2']
B_equipment = ['B1', 'B2', 'B3']
eligibility = {'I': {'A': ['A1', 'A2'], 'B': ['B1', 'B2', 'B3']}, 'II': {'A': ['A1', 'A2'], 'B': ['B1']}, 'III': {'A': ['A2'], 'B': ['B2']}}

def find_row_idx(label):
    idx = df['Equipment / Cost'].str.strip().str.casefold() == label.strip().casefold()
    matches = np.flatnonzero(idx.values)
    if len(matches) == 0:
        raise ValueError(f"Row '{label}' not found in CSV.")
    return matches[0]

def get_numeric(row, col):
    val = df.at[row, col]
    if val == '':
        raise ValueError(f'Missing value for {col} at row {row}')
    return float(val)
processing_time = {}
equipment_procedure = {}
for eq in A_equipment:
    equipment_procedure[eq] = 'A'
for eq in B_equipment:
    equipment_procedure[eq] = 'B'
for p_full in products:
    p = product_short[p_full]
    for proc in procedures:
        eq_list = eligibility[p][proc]
        for eq in eq_list:
            eq_row_idx = find_row_idx(eq)
            val = df.at[eq_row_idx, p_full]
            if val == '':
                raise ValueError(f'Missing processing time for {p_full} on {eq}')
            processing_time[p, proc, eq] = float(val)
equipment_available_time = {}
equipment_full_load_cost = {}
for eq in A_equipment + B_equipment:
    eq_row_idx = find_row_idx(eq)
    avail_time_str = df.at[eq_row_idx, 'Available Equipment Operating Time']
    if avail_time_str == '':
        raise ValueError(f'Missing available operating time for {eq}')
    equipment_available_time[eq] = float(avail_time_str)
    cost_str = df.at[eq_row_idx, 'Equipment Cost at Full Load (yuan)']
    if cost_str == '':
        raise ValueError(f'Missing equipment cost at full load for {eq}')
    equipment_full_load_cost[eq] = float(cost_str)
raw_material_cost_row = find_row_idx('Raw Material Cost (yuan/unit)')
unit_price_row = find_row_idx('Unit Price (yuan/unit)')
raw_material_cost = {}
unit_price = {}
for p_full in products:
    p = product_short[p_full]
    val = df.at[raw_material_cost_row, p_full]
    if val == '':
        raise ValueError(f'Missing raw material cost for {p_full}')
    raw_material_cost[p] = float(val)
    val = df.at[unit_price_row, p_full]
    if val == '':
        raise ValueError(f'Missing unit price for {p_full}')
    unit_price[p] = float(val)

def solve_problem():
    m = gp.Model('FactoryProductionPlan')
    x_keys = []
    for p in ['I', 'II', 'III']:
        for proc in procedures:
            for eq in eligibility[p][proc]:
                x_keys.append((p, proc, eq))
    quantity_vars = m.addVars(x_keys, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    total_produced_A = {}
    total_produced_B = {}
    for p in ['I', 'II', 'III']:
        total_produced_A[p] = gp.quicksum((quantity_vars[p, 'A', eq] for eq in eligibility[p]['A']))
        total_produced_B[p] = gp.quicksum((quantity_vars[p, 'B', eq] for eq in eligibility[p]['B']))
    for p in ['I', 'II', 'III']:
        m.addConstr(total_produced_A[p] == total_produced_B[p], name=f'proc_sync_{p}')
    for eq in A_equipment + B_equipment:
        used_time = gp.LinExpr()
        for p in ['I', 'II', 'III']:
            proc = equipment_procedure[eq]
            if eq in eligibility[p][proc]:
                used_time += processing_time[p, proc, eq] * quantity_vars[p, proc, eq]
        m.addConstr(used_time <= equipment_available_time[eq], name=f'eq_time_{eq}')
    total_revenue = gp.quicksum((unit_price[p] * total_produced_A[p] for p in ['I', 'II', 'III']))
    total_raw_material_cost = gp.quicksum((raw_material_cost[p] * total_produced_A[p] for p in ['I', 'II', 'III']))
    total_equipment_cost = gp.LinExpr()
    for eq in A_equipment + B_equipment:
        used_time = gp.LinExpr()
        for p in ['I', 'II', 'III']:
            proc = equipment_procedure[eq]
            if eq in eligibility[p][proc]:
                used_time += processing_time[p, proc, eq] * quantity_vars[p, proc, eq]
        total_equipment_cost += equipment_full_load_cost[eq] * (used_time / equipment_available_time[eq])
    total_profit = total_revenue - total_raw_material_cost - total_equipment_cost
    m.setObjective(total_profit, gp.GRB.MAXIMIZE)
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')