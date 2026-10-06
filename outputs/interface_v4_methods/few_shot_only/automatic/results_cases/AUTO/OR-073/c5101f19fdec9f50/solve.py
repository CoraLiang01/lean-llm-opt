import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv'
df = pd.read_csv(csv_path, sep=',')
products = ['Product I', 'Product II', 'Product III']
product_short = {'Product I': 'I', 'Product II': 'II', 'Product III': 'III'}
procedures = ['A', 'B']
A_equipment = ['A1', 'A2']
B_equipment = ['B1', 'B2', 'B3']
equipment_list = A_equipment + B_equipment
equipment_procedure = {}
for eq in A_equipment:
    equipment_procedure[eq] = 'A'
for eq in B_equipment:
    equipment_procedure[eq] = 'B'

def get_row_idx(label):
    idx = df.index[df['Equipment / Cost'].astype(str).str.strip().str.casefold() == label.strip().casefold()]
    if len(idx) == 0:
        raise KeyError(f"Row '{label}' not found in CSV.")
    return idx[0]
processing_time = {}
for eq in equipment_list:
    try:
        row = df.loc[get_row_idx(eq)]
    except KeyError:
        continue
    for prod in products:
        val = row[prod]
        if pd.notnull(val):
            processing_time[prod, eq] = float(val)
equipment_available_time = {}
equipment_full_load_cost = {}
for eq in equipment_list:
    try:
        row = df.loc[get_row_idx(eq)]
    except KeyError:
        continue
    avail = row['Available Equipment Operating Time']
    if pd.notnull(avail):
        equipment_available_time[eq] = float(avail)
    cost = row['Equipment Cost at Full Load (yuan)']
    if pd.notnull(cost):
        equipment_full_load_cost[eq] = float(cost)
raw_mat_row = get_row_idx('Raw Material Cost (yuan/unit)')
raw_material_cost = {}
for prod in products:
    val = df.loc[raw_mat_row, prod]
    if pd.notnull(val):
        raw_material_cost[prod] = float(val)
    else:
        raise ValueError(f'Missing raw material cost for {prod}')
unit_price_row = get_row_idx('Unit Price (yuan/unit)')
selling_price = {}
for prod in products:
    val = df.loc[unit_price_row, prod]
    if pd.notnull(val):
        selling_price[prod] = float(val)
    else:
        raise ValueError(f'Missing selling price for {prod}')
compatible_equipment = {'Product I': {'A': ['A1', 'A2'], 'B': ['B1', 'B2', 'B3']}, 'Product II': {'A': ['A1', 'A2'], 'B': ['B1']}, 'Product III': {'A': ['A2'], 'B': ['B2']}}
m = gp.Model('FactoryProductionPlan')
x_vars = {}
for prod in products:
    for proc in procedures:
        for eq in compatible_equipment[prod][proc]:
            if (prod, eq) in processing_time:
                x_vars[prod, eq] = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name=f'x_{product_short[prod]}_{eq}')
for prod in products:
    sum_A = gp.quicksum((x_vars[prod, eq] for eq in compatible_equipment[prod]['A'] if (prod, eq) in x_vars))
    sum_B = gp.quicksum((x_vars[prod, eq] for eq in compatible_equipment[prod]['B'] if (prod, eq) in x_vars))
    m.addConstr(sum_A == sum_B, name=f'proc_balance_{product_short[prod]}')
for eq in equipment_list:
    relevant_prods = [prod for prod in products if (prod, eq) in x_vars]
    if not relevant_prods:
        continue
    total_time = gp.quicksum((processing_time[prod, eq] * x_vars[prod, eq] for prod in relevant_prods))
    if eq not in equipment_available_time:
        raise ValueError(f'Missing available operating time for equipment {eq}')
    m.addConstr(total_time <= equipment_available_time[eq], name=f'time_limit_{eq}')
total_produced = {}
for prod in products:
    total_produced[prod] = gp.quicksum((x_vars[prod, eq] for eq in compatible_equipment[prod]['A'] if (prod, eq) in x_vars))
revenue = gp.quicksum((selling_price[prod] * total_produced[prod] for prod in products))
raw_mat_cost = gp.quicksum((raw_material_cost[prod] * total_produced[prod] for prod in products))
equipment_cost = 0
for eq in equipment_list:
    relevant_prods = [prod for prod in products if (prod, eq) in x_vars]
    if not relevant_prods:
        continue
    used_time = gp.quicksum((processing_time[prod, eq] * x_vars[prod, eq] for prod in relevant_prods))
    if eq not in equipment_available_time or eq not in equipment_full_load_cost:
        raise ValueError(f'Missing equipment time or cost for {eq}')
    equipment_cost += equipment_full_load_cost[eq] * (used_time / equipment_available_time[eq])
m.setObjective(revenue - raw_mat_cost - equipment_cost, gp.GRB.MAXIMIZE)
m.optimize()