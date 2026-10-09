import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv'
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
products = ['Product I', 'Product II', 'Product III']
procedures = ['A', 'B']
equipment_rows = df[~df['Equipment / Cost'].str.strip().str.casefold().isin(['raw material cost (yuan/unit)', 'unit price (yuan/unit)'])].copy()
equipment_rows = equipment_rows[(equipment_rows['Available Equipment Operating Time'].str.strip() != '') | (equipment_rows['Equipment Cost at Full Load (yuan)'].str.strip() != '')].copy()
equipment_list = equipment_rows['Equipment / Cost'].tolist()
eligibility = {('Product I', 'A'): ['A1', 'A2'], ('Product I', 'B'): ['B1', 'B2', 'B3'], ('Product II', 'A'): ['A1', 'A2'], ('Product II', 'B'): ['B1'], ('Product III', 'A'): ['A2'], ('Product III', 'B'): ['B2']}
processing_time = {}
for (idx, row) in equipment_rows.iterrows():
    equip = row['Equipment / Cost'].strip()
    for prod in products:
        val = row[prod].strip()
        if val != '':
            try:
                processing_time[prod, equip] = float(val)
            except Exception:
                raise ValueError(f"Invalid processing time for ({prod}, {equip}): '{val}'")
equipment_operating_time = {}
equipment_full_load_cost = {}
for (idx, row) in equipment_rows.iterrows():
    equip = row['Equipment / Cost'].strip()
    avail_time = row['Available Equipment Operating Time'].strip()
    cost_full_load = row['Equipment Cost at Full Load (yuan)'].strip()
    if avail_time != '':
        try:
            equipment_operating_time[equip] = float(avail_time)
        except Exception:
            raise ValueError(f"Invalid available operating time for {equip}: '{avail_time}'")
    if cost_full_load != '':
        try:
            equipment_full_load_cost[equip] = float(cost_full_load)
        except Exception:
            raise ValueError(f"Invalid equipment cost at full load for {equip}: '{cost_full_load}'")

def get_param_row(param_name):
    mask = df['Equipment / Cost'].str.strip().str.casefold() == param_name.strip().casefold()
    if not mask.any():
        raise ValueError(f"Parameter row '{param_name}' not found in CSV.")
    return df[mask].iloc[0]
raw_material_cost_row = get_param_row('Raw Material Cost (yuan/unit)')
unit_price_row = get_param_row('Unit Price (yuan/unit)')
raw_material_cost = {}
unit_price = {}
for prod in products:
    val_rm = raw_material_cost_row[prod].strip()
    val_up = unit_price_row[prod].strip()
    if val_rm == '' or val_up == '':
        raise ValueError(f'Missing raw material cost or unit price for {prod}')
    try:
        raw_material_cost[prod] = float(val_rm)
        unit_price[prod] = float(val_up)
    except Exception:
        raise ValueError(f"Invalid raw material cost or unit price for {prod}: '{val_rm}', '{val_up}'")
decision_tuples = []
for prod in products:
    for proc in procedures:
        for equip in eligibility.get((prod, proc), []):
            decision_tuples.append((prod, proc, equip))
m = gp.Model('ProductionPlan')
x_vars = m.addVars(decision_tuples, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
for prod in products:
    sum_A = gp.quicksum((x_vars[prod, 'A', equip] for equip in eligibility.get((prod, 'A'), [])))
    sum_B = gp.quicksum((x_vars[prod, 'B', equip] for equip in eligibility.get((prod, 'B'), [])))
    m.addConstr(sum_A == sum_B, name=f'proc_balance_{prod}')
for equip in equipment_list:
    relevant_vars = []
    for prod in products:
        for proc in procedures:
            if (prod, proc, equip) in x_vars:
                pt = processing_time.get((prod, equip), None)
                if pt is None:
                    raise ValueError(f'Missing processing time for ({prod}, {equip})')
                relevant_vars.append((x_vars[prod, proc, equip], pt))
    if equip in equipment_operating_time:
        m.addConstr(gp.quicksum((var * pt for (var, pt) in relevant_vars)) <= equipment_operating_time[equip], name=f'equip_capacity_{equip}')
total_units = {}
for prod in products:
    total_units[prod] = gp.quicksum((x_vars[prod, 'A', equip] for equip in eligibility.get((prod, 'A'), [])))
total_revenue = gp.quicksum((unit_price[prod] * total_units[prod] for prod in products))
total_raw_material_cost = gp.quicksum((raw_material_cost[prod] * total_units[prod] for prod in products))
total_equipment_cost_terms = []
for equip in equipment_list:
    if equip in equipment_operating_time and equip in equipment_full_load_cost:
        total_time_used = gp.quicksum((x_vars[prod, proc, equip] * processing_time[prod, equip] for prod in products for proc in procedures if (prod, proc, equip) in x_vars))
        equip_cost = equipment_full_load_cost[equip] * (total_time_used / equipment_operating_time[equip])
        total_equipment_cost_terms.append(equip_cost)
total_equipment_cost = gp.quicksum(total_equipment_cost_terms)
m.setObjective(total_revenue - total_raw_material_cost - total_equipment_cost, gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total profit: {m.objVal:.2f}\n')
    print('--- Production Plan ---')
    for prod in products:
        qty = total_units[prod].getValue()
        print(f'{prod}: {qty:.2f} units produced')
    print('\n--- Assignment Details ---')
    for (key, var) in x_vars.items():
        if var.X > 1e-06:
            (prod, proc, equip) = key
            print(f'{prod}, Procedure {proc}, Equipment {equip}: {var.X:.2f} units')
    print('\n--- Equipment Utilization ---')
    for equip in equipment_list:
        if equip in equipment_operating_time and equip in equipment_full_load_cost:
            total_time = sum((x_vars[prod, proc, equip].X * processing_time[prod, equip] for prod in products for proc in procedures if (prod, proc, equip) in x_vars))
            utilization = total_time / equipment_operating_time[equip] if equipment_operating_time[equip] > 0 else 0
            print(f'{equip}: Used {total_time:.2f} hours ({utilization:.1%} of available)')
else:
    print(f'No optimal solution found. Status: {m.status}')