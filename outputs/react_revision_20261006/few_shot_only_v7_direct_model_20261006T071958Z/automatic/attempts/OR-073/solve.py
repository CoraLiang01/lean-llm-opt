import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
products = ['Product I', 'Product II', 'Product III']
product_short = {'Product I': 'I', 'Product II': 'II', 'Product III': 'III'}
procedures = ['A', 'B']
equipments_A = ['A1', 'A2']
equipments_B = ['B1', 'B2', 'B3']
equipments = equipments_A + equipments_B
eligible_equipment = {'A': {'Product I': ['A1', 'A2'], 'Product II': ['A1', 'A2'], 'Product III': ['A2']}, 'B': {'Product I': ['B1', 'B2', 'B3'], 'Product II': ['B1'], 'Product III': ['B2']}}

def get_row(label):
    idx = df['Equipment / Cost'].str.casefold() == label.casefold()
    if not idx.any():
        raise ValueError(f"Row '{label}' not found in CSV.")
    return df[idx].iloc[0]
processing_time = {}
for eq in equipments:
    row = df[df['Equipment / Cost'].str.casefold() == eq.casefold()]
    if row.empty:
        raise ValueError(f"Equipment '{eq}' not found in CSV.")
    row = row.iloc[0]
    for prod in products:
        val = row[prod]
        try:
            pt = float(val)
        except Exception:
            raise ValueError(f"Processing time missing or invalid for equipment '{eq}', product '{prod}'.")
        processing_time[eq, prod] = pt
raw_material_row = get_row('Raw Material Cost (yuan/unit)')
raw_material_cost = {}
for prod in products:
    try:
        raw_material_cost[prod] = float(raw_material_row[prod])
    except Exception:
        raise ValueError(f"Raw material cost missing or invalid for product '{prod}'.")
unit_price_row = get_row('Unit Price (yuan/unit)')
selling_price = {}
for prod in products:
    try:
        selling_price[prod] = float(unit_price_row[prod])
    except Exception:
        raise ValueError(f"Selling price missing or invalid for product '{prod}'.")
available_time = {}
for eq in equipments:
    row = df[df['Equipment / Cost'].str.casefold() == eq.casefold()]
    if row.empty:
        raise ValueError(f"Equipment '{eq}' not found in CSV.")
    val = row['Available Equipment Operating Time']
    try:
        available_time[eq] = float(val)
    except Exception:
        raise ValueError(f"Available equipment operating time missing or invalid for equipment '{eq}'.")
equipment_cost = {}
for eq in equipments:
    row = df[df['Equipment / Cost'].str.casefold() == eq.casefold()]
    if row.empty:
        raise ValueError(f"Equipment '{eq}' not found in CSV.")
    val = row['Equipment Cost at Full Load (yuan)']
    try:
        equipment_cost[eq] = float(val)
    except Exception:
        raise ValueError(f"Equipment cost at full load missing or invalid for equipment '{eq}'.")

def solve_problem():
    m = gp.Model('FactoryProductionPlan')
    x_keys = []
    for proc in procedures:
        for prod in products:
            for eq in eligible_equipment[proc][prod]:
                x_keys.append((prod, proc, eq))
    x_vars = m.addVars(x_keys, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    total_produced_A = {}
    total_produced_B = {}
    for prod in products:
        total_produced_A[prod] = gp.quicksum((x_vars[prod, 'A', eq] for eq in eligible_equipment['A'][prod]))
        total_produced_B[prod] = gp.quicksum((x_vars[prod, 'B', eq] for eq in eligible_equipment['B'][prod]))
    for prod in products:
        m.addConstr(total_produced_A[prod] == total_produced_B[prod], name=f'consistency_{product_short[prod]}')
    for eq in equipments:
        relevant_x = []
        for proc in procedures:
            for prod in products:
                if eq in eligible_equipment[proc][prod]:
                    relevant_x.append((prod, proc, eq))
        expr = gp.quicksum((x_vars[key] * processing_time[eq, key[0]] for key in relevant_x))
        m.addConstr(expr <= available_time[eq], name=f'timelimit_{eq}')
    total_revenue = gp.quicksum((selling_price[prod] * total_produced_A[prod] for prod in products))
    total_raw_material_cost = gp.quicksum((raw_material_cost[prod] * total_produced_A[prod] for prod in products))
    total_equipment_cost = gp.quicksum((equipment_cost[eq] * (gp.quicksum((x_vars[prod, proc, eq] * processing_time[eq, prod] for proc in procedures for prod in products if (prod, proc, eq) in x_vars)) / available_time[eq]) for eq in equipments))
    m.setObjective(total_revenue - total_raw_material_cost - total_equipment_cost, gp.GRB.MAXIMIZE)
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