import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv'
df = pd.read_csv(csv_path, sep=',')
products = ['Product I', 'Product II', 'Product III']
product_short = {'Product I': 'I', 'Product II': 'II', 'Product III': 'III'}
procedures = ['A', 'B']
equip_A = ['A1', 'A2']
equip_B = ['B1', 'B2', 'B3']
equip_proc = {}
for eq in equip_A:
    equip_proc[eq] = 'A'
for eq in equip_B:
    equip_proc[eq] = 'B'
eligible_equipment = {('Product I', 'A'): ['A1', 'A2'], ('Product I', 'B'): ['B1', 'B2', 'B3'], ('Product II', 'A'): ['A1', 'A2'], ('Product II', 'B'): ['B1'], ('Product III', 'A'): ['A2'], ('Product III', 'B'): ['B2']}

def get_row_idx(label):
    idx = df['Equipment / Cost'].str.strip().str.casefold() == label.strip().casefold()
    matches = np.where(idx)[0]
    if len(matches) == 0:
        raise KeyError(f"Row '{label}' not found in 'Equipment / Cost'")
    return matches[0]
all_equipment = set(sum([v for v in eligible_equipment.values()], []))
equipment_rows = df['Equipment / Cost'].str.strip().isin(all_equipment)
equip_df = df.loc[equipment_rows].copy()
equip_df['Equipment'] = equip_df['Equipment / Cost'].str.strip()
equip_available_time = {}
equip_full_load_cost = {}
for (_, row) in equip_df.iterrows():
    eq = row['Equipment']
    if pd.isna(row['Available Equipment Operating Time']):
        raise ValueError(f'Missing available time for equipment {eq}')
    if pd.isna(row['Equipment Cost at Full Load (yuan)']):
        raise ValueError(f'Missing full load cost for equipment {eq}')
    equip_available_time[eq] = float(row['Available Equipment Operating Time'])
    equip_full_load_cost[eq] = float(row['Equipment Cost at Full Load (yuan)'])
proc_time = {}
for p in products:
    for proc in procedures:
        for eq in eligible_equipment[p, proc]:
            eq_row = equip_df[equip_df['Equipment'] == eq]
            if eq_row.empty:
                raise KeyError(f'Equipment {eq} not found in CSV for product {p}')
            val = eq_row.iloc[0][p]
            if pd.isna(val):
                raise ValueError(f'Missing processing time for product {p} on equipment {eq}')
            proc_time[p, eq] = float(val)
rmc_idx = get_row_idx('Raw Material Cost (yuan/unit)')
raw_material_cost = {}
for p in products:
    val = df.loc[rmc_idx, p]
    if pd.isna(val):
        raise ValueError(f'Missing raw material cost for {p}')
    raw_material_cost[p] = float(val)
usp_idx = get_row_idx('Unit Price (yuan/unit)')
unit_price = {}
for p in products:
    val = df.loc[usp_idx, p]
    if pd.isna(val):
        raise ValueError(f'Missing unit price for {p}')
    unit_price[p] = float(val)
x_keys = []
for p in products:
    for proc in procedures:
        for eq in eligible_equipment[p, proc]:
            x_keys.append((p, eq))
m = gp.Model('ProductionPlan')
x = m.addVars(x_keys, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
for eq in all_equipment:
    relevant_x = [(p, eq) for (p, e) in x_keys if e == eq]
    expr = gp.quicksum((proc_time[p, eq] * x[p, eq] for (p, eq) in relevant_x))
    m.addConstr(expr <= equip_available_time[eq], name=f'time_{eq}')
for p in products:
    sum_A = gp.quicksum((x[p, eq] for eq in eligible_equipment[p, 'A']))
    sum_B = gp.quicksum((x[p, eq] for eq in eligible_equipment[p, 'B']))
    m.addConstr(sum_A == sum_B, name=f'flow_{product_short[p]}')
total_revenue = gp.quicksum((unit_price[p] * gp.quicksum((x[p, eq] for eq in eligible_equipment[p, 'B'])) for p in products))
total_rm_cost = gp.quicksum((raw_material_cost[p] * gp.quicksum((x[p, eq] for eq in eligible_equipment[p, 'B'])) for p in products))
equip_operating_cost = gp.quicksum((equip_full_load_cost[eq] * (gp.quicksum((proc_time[p, eq] * x[p, eq] for (p, e) in x_keys if e == eq)) / equip_available_time[eq]) for eq in all_equipment))
m.setObjective(total_revenue - total_rm_cost - equip_operating_cost, gp.GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal:.6f}')
    for (p, eq) in x_keys:
        print(f'{x[p, eq].VarName}: {x[p, eq].X:.6f}')
else:
    print(f'Solver status: {m.status}')