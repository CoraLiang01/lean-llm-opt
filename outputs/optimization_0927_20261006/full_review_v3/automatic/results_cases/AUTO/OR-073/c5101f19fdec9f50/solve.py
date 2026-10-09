import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
products = ['Product I', 'Product II', 'Product III']
product_short = {'Product I': 'I', 'Product II': 'II', 'Product III': 'III'}
procedures = ['A', 'B']
equipment_rows = df[(df['Available Equipment Operating Time'].str.strip() != '') & (df['Equipment Cost at Full Load (yuan)'].str.strip() != '')].copy()
equipment_ids = equipment_rows['Equipment / Cost'].tolist()
equipment_proc = {}
for eq in equipment_ids:
    eq_clean = eq.strip().upper()
    if eq_clean.startswith('A'):
        equipment_proc[eq] = 'A'
    elif eq_clean.startswith('B'):
        equipment_proc[eq] = 'B'
    else:
        raise ValueError(f'Unknown equipment type: {eq}')
eligible_equipment = {('Product I', 'A'): [e for e in equipment_ids if equipment_proc[e] == 'A' and e.strip().upper() in ['A1', 'A2']], ('Product I', 'B'): [e for e in equipment_ids if equipment_proc[e] == 'B' and e.strip().upper() in ['B1', 'B2', 'B3']], ('Product II', 'A'): [e for e in equipment_ids if equipment_proc[e] == 'A' and e.strip().upper() in ['A1', 'A2']], ('Product II', 'B'): [e for e in equipment_ids if equipment_proc[e] == 'B' and e.strip().upper() == 'B1'], ('Product III', 'A'): [e for e in equipment_ids if equipment_proc[e] == 'A' and e.strip().upper() == 'A2'], ('Product III', 'B'): [e for e in equipment_ids if equipment_proc[e] == 'B' and e.strip().upper() == 'B2']}
for col in products + ['Available Equipment Operating Time', 'Equipment Cost at Full Load (yuan)']:
    equipment_rows[col] = pd.to_numeric(equipment_rows[col], errors='coerce')
processing_time = {}
for (_, row) in equipment_rows.iterrows():
    eq = row['Equipment / Cost']
    for p in products:
        val = row[p]
        if not np.isnan(val):
            processing_time[p, eq] = float(val)
equipment_operating_time = equipment_rows.set_index('Equipment / Cost')['Available Equipment Operating Time'].to_dict()
equipment_operating_time = {k: float(v) for (k, v) in equipment_operating_time.items()}
equipment_cost_full_load = equipment_rows.set_index('Equipment / Cost')['Equipment Cost at Full Load (yuan)'].to_dict()
equipment_cost_full_load = {k: float(v) for (k, v) in equipment_cost_full_load.items()}

def find_row_idx(label):
    return df['Equipment / Cost'].str.strip().str.casefold() == label.strip().casefold()
raw_material_row = df[find_row_idx('Raw Material Cost (yuan/unit)')].iloc[0]
unit_price_row = df[find_row_idx('Unit Price (yuan/unit)')].iloc[0]
raw_material_cost = {p: float(raw_material_row[p]) for p in products}
unit_price = {p: float(unit_price_row[p]) for p in products}
x_vars = {}
for p in products:
    for proc in procedures:
        for eq in eligible_equipment[p, proc]:
            x_vars[p, proc, eq] = None
m = gp.Model('FactoryProductionPlan')
for key in x_vars:
    (p, proc, eq) = key
    x_vars[key] = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name=f'x_{product_short[p]}_{proc}_{eq}')
m.update()
prod_total_vars = {}
for p in products:
    prod_total_vars[p] = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name=f'q_{product_short[p]}')
m.update()
for p in products:
    m.addConstr(gp.quicksum((x_vars[p, 'A', eq] for eq in eligible_equipment[p, 'A'])) == prod_total_vars[p], name=f'procA_balance_{product_short[p]}')
    m.addConstr(gp.quicksum((x_vars[p, 'B', eq] for eq in eligible_equipment[p, 'B'])) == prod_total_vars[p], name=f'procB_balance_{product_short[p]}')
for eq in equipment_ids:
    relevant_keys = [k for k in x_vars if k[2] == eq]
    m.addConstr(gp.quicksum((processing_time[k[0], eq] * x_vars[k] for k in relevant_keys)) <= equipment_operating_time[eq], name=f'time_limit_{eq}')
total_revenue = gp.quicksum((unit_price[p] * prod_total_vars[p] for p in products))
total_raw_material_cost = gp.quicksum((raw_material_cost[p] * prod_total_vars[p] for p in products))
equipment_usage_time = {}
for eq in equipment_ids:
    relevant_keys = [k for k in x_vars if k[2] == eq]
    equipment_usage_time[eq] = gp.quicksum((processing_time[k[0], eq] * x_vars[k] for k in relevant_keys))
total_equipment_cost = gp.quicksum((equipment_cost_full_load[eq] * (equipment_usage_time[eq] / equipment_operating_time[eq]) for eq in equipment_ids))
m.setObjective(total_revenue - total_raw_material_cost - total_equipment_cost, gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total profit: {m.objVal:.2f}\n')
    print('--- Production Plan ---')
    for p in products:
        q = prod_total_vars[p].X
        print(f'Product {product_short[p]}: {q:.2f} units')
        for proc in procedures:
            for eq in eligible_equipment[p, proc]:
                v = x_vars[p, proc, eq].X
                if v > 1e-06:
                    print(f'  {proc} on {eq}: {v:.2f} units')
    print('\n--- Equipment Usage ---')
    for eq in equipment_ids:
        used_time = equipment_usage_time[eq].getValue()
        print(f'{eq}: Used {used_time:.2f} / {equipment_operating_time[eq]:.2f} hours, Cost: {equipment_cost_full_load[eq] * (used_time / equipment_operating_time[eq]):.2f}')
    print('\n--- Cost Breakdown ---')
    print(f'Total revenue: {total_revenue.getValue():.2f}')
    print(f'Total raw material cost: {total_raw_material_cost.getValue():.2f}')
    print(f'Total equipment cost: {total_equipment_cost.getValue():.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')