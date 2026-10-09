import gurobipy as gp
import pandas as pd
import numpy as np
import re

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv'
    df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
    products = ['I', 'II', 'III']
    procedures = ['A', 'B']
    equip_A = ['A1', 'A2']
    equip_B = ['B1', 'B2', 'B3']
    eligible_equipment = {('I', 'A'): ['A1', 'A2'], ('I', 'B'): ['B1', 'B2', 'B3'], ('II', 'A'): ['A1', 'A2'], ('II', 'B'): ['B1'], ('III', 'A'): ['A2'], ('III', 'B'): ['B2']}

    def find_row(label):
        idx = df['Equipment / Cost'].str.strip().str.casefold() == label.strip().casefold()
        matches = df[idx]
        if matches.shape[0] != 1:
            raise ValueError(f"Expected exactly one row for '{label}', found {matches.shape[0]}")
        return matches.index[0]
    equipment_rows = []
    for (idx, val) in df['Equipment / Cost'].items():
        v = val.strip()
        if re.fullmatch('[AB]\\d+', v):
            equipment_rows.append(idx)
    equipment_list = []
    for idx in equipment_rows:
        equipment_list.append(df.at[idx, 'Equipment / Cost'].strip())
    equipment_to_row = {df.at[idx, 'Equipment / Cost'].strip(): idx for idx in equipment_rows}
    available_time = {}
    full_load_cost = {}
    for eq in equipment_list:
        idx = equipment_to_row[eq]
        avail_time_str = df.at[idx, 'Available Equipment Operating Time'].strip()
        full_cost_str = df.at[idx, 'Equipment Cost at Full Load (yuan)'].strip()
        if avail_time_str == '' or full_cost_str == '':
            raise ValueError(f'Missing available time or full load cost for equipment {eq}')
        available_time[eq] = float(avail_time_str)
        full_load_cost[eq] = float(full_cost_str)
    processing_time = {}
    product_col_map = {'I': 'Product I', 'II': 'Product II', 'III': 'Product III'}
    for eq in equipment_list:
        idx = equipment_to_row[eq]
        for p in products:
            col = product_col_map[p]
            val = df.at[idx, col].strip()
            if val != '':
                processing_time[p, eq] = float(val)
    idx_rm = find_row('Raw Material Cost (yuan/unit)')
    raw_material_cost = {}
    for p in products:
        col = product_col_map[p]
        val = df.at[idx_rm, col].strip()
        if val == '':
            raise ValueError(f'Missing raw material cost for product {p}')
        raw_material_cost[p] = float(val)
    idx_price = find_row('Unit Price (yuan/unit)')
    unit_price = {}
    for p in products:
        col = product_col_map[p]
        val = df.at[idx_price, col].strip()
        if val == '':
            raise ValueError(f'Missing unit price for product {p}')
        unit_price[p] = float(val)
    decision_keys = []
    for p in products:
        for proc in procedures:
            for eq in eligible_equipment.get((p, proc), []):
                if (p, eq) in processing_time:
                    decision_keys.append((p, proc, eq))
    for p in products:
        for proc in procedures:
            for eq in eligible_equipment.get((p, proc), []):
                if (p, eq) not in processing_time:
                    raise ValueError(f'Missing processing time for product {p} on equipment {eq}')
    m = gp.Model('ProductionPlan')
    x_vars = m.addVars(decision_keys, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    for eq in equipment_list:
        expr = gp.LinExpr()
        for (p, proc, eq2) in decision_keys:
            if eq2 == eq:
                expr += processing_time[p, eq] * x_vars[p, proc, eq]
        m.addConstr(expr <= available_time[eq], name=f'time_{eq}')
    for p in products:
        sum_A = gp.quicksum((x_vars[p, 'A', eq] for eq in eligible_equipment.get((p, 'A'), []) if (p, 'A', eq) in x_vars))
        sum_B = gp.quicksum((x_vars[p, 'B', eq] for eq in eligible_equipment.get((p, 'B'), []) if (p, 'B', eq) in x_vars))
        m.addConstr(sum_A == sum_B, name=f'consistency_{p}')
    total_produced = {}
    for p in products:
        total_produced[p] = gp.quicksum((x_vars[p, 'A', eq] for eq in eligible_equipment.get((p, 'A'), []) if (p, 'A', eq) in x_vars))
    total_revenue = gp.quicksum((unit_price[p] * total_produced[p] for p in products))
    total_raw_material_cost = gp.quicksum((raw_material_cost[p] * total_produced[p] for p in products))
    equipment_operating_cost = gp.LinExpr()
    for eq in equipment_list:
        used_time = gp.LinExpr()
        for (p, proc, eq2) in decision_keys:
            if eq2 == eq:
                used_time += processing_time[p, eq] * x_vars[p, proc, eq]
        equipment_operating_cost += used_time / available_time[eq] * full_load_cost[eq]
    total_profit = total_revenue - total_raw_material_cost - equipment_operating_cost
    m.setObjective(total_profit, gp.GRB.MAXIMIZE)
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal:.6f}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X:.6f}')
else:
    print(f'Solver status: {m.status}')