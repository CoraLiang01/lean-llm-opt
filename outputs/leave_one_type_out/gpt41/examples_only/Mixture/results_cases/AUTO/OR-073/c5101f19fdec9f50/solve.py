import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv'
df = pd.read_csv(csv_path, sep=',')
products = ['Product I', 'Product II', 'Product III']
product_short = {'Product I': 'I', 'Product II': 'II', 'Product III': 'III'}
procedures = ['A', 'B']
equip_A = ['A1', 'A2']
equip_B = ['B1', 'B2', 'B3']
equipments = equip_A + equip_B
equip_to_proc = {}
for e in equip_A:
    equip_to_proc[e] = 'A'
for e in equip_B:
    equip_to_proc[e] = 'B'
equipment_rows = df['Equipment / Cost'].str.match('^(A1|A2|B1|B2|B3)$', case=False, na=False)
equip_df = df.loc[equipment_rows].copy()
equip_df['Equipment / Cost'] = equip_df['Equipment / Cost'].str.strip()
equip_df.set_index('Equipment / Cost', inplace=True)
avail_time = {}
equip_full_cost = {}
for e in equipments:
    if e not in equip_df.index:
        raise ValueError(f'Equipment {e} not found in CSV.')
    avail_time[e] = float(equip_df.loc[e, 'Available Equipment Operating Time'])
    equip_full_cost[e] = float(equip_df.loc[e, 'Equipment Cost at Full Load (yuan)'])
proc_time = {}
for e in equipments:
    proc_time[e] = {}
    for p in products:
        val = equip_df.loc[e, p]
        if pd.isna(val):
            proc_time[e][p] = 0.0
        else:
            proc_time[e][p] = float(val)

def get_row_value(row_name, col):
    row = df['Equipment / Cost'].str.strip().str.casefold() == row_name.casefold()
    if not row.any():
        raise ValueError(f"Row '{row_name}' not found in CSV.")
    val = df.loc[row, col].values[0]
    if pd.isna(val):
        raise ValueError(f"Missing value for '{row_name}', '{col}'")
    return float(val)
raw_mat_cost = {}
unit_price = {}
for p in products:
    raw_mat_cost[p] = get_row_value('Raw Material Cost (yuan/unit)', p)
    unit_price[p] = get_row_value('Unit Price (yuan/unit)', p)
allowed_equip = {}
for p in products:
    allowed_equip[p] = {}
allowed_equip['Product I']['A'] = ['A1', 'A2']
allowed_equip['Product I']['B'] = ['B1', 'B2', 'B3']
allowed_equip['Product II']['A'] = ['A1', 'A2']
allowed_equip['Product II']['B'] = ['B1']
allowed_equip['Product III']['A'] = ['A2']
allowed_equip['Product III']['B'] = ['B2']
m = gp.Model('FactoryProductionPlan')
x = m.addVars(products, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y_vars = {}
for p in products:
    for proc in procedures:
        for e in allowed_equip[p][proc]:
            y_vars[e, p] = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name=f'y_{e}_{product_short[p]}')
for p in products:
    for proc in procedures:
        eqs = [y_vars[e, p] for e in allowed_equip[p][proc]]
        m.addConstr(gp.quicksum(eqs) == x[p], name=f'assign_{product_short[p]}_{proc}')
for e in equipments:
    time_expr = []
    for p in products:
        if (e, p) in y_vars:
            time_per_unit = proc_time[e][p]
            time_expr.append(time_per_unit * y_vars[e, p])
    if time_expr:
        m.addConstr(gp.quicksum(time_expr) <= avail_time[e], name=f'time_{e}')
revenue = gp.quicksum((unit_price[p] * x[p] for p in products))
raw_cost = gp.quicksum((raw_mat_cost[p] * x[p] for p in products))
equip_cost_expr = []
for e in equipments:
    time_used = gp.quicksum((proc_time[e][p] * y_vars[e, p] for p in products if (e, p) in y_vars))
    equip_cost_expr.append(equip_full_cost[e] * (time_used / avail_time[e]))
equip_cost = gp.quicksum(equip_cost_expr)
m.setObjective(revenue - raw_cost - equip_cost, gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total profit: {m.objVal:.2f}')
    print('\n--- Production Quantities ---')
    for p in products:
        print(f'  {p}: {x[p].X:.4f} units')
    print('\n--- Equipment Assignment (units processed) ---')
    for (e, p), var in y_vars.items():
        if var.X > 1e-06:
            print(f'  {e} processes {var.X:.4f} units of {p}')
    print('\n--- Equipment Utilization ---')
    for e in equipments:
        used = sum((proc_time[e][p] * y_vars[e, p].X for p in products if (e, p) in y_vars))
        print(f'  {e}: {used:.2f} / {avail_time[e]:.2f} hours used ({100 * used / avail_time[e]:.2f}%)')
    print('\n--- Cost Breakdown ---')
    print(f'  Total revenue: {sum((unit_price[p] * x[p].X for p in products)):.2f}')
    print(f'  Raw material cost: {sum((raw_mat_cost[p] * x[p].X for p in products)):.2f}')
    print(f'  Equipment cost: {sum((equip_full_cost[e] * (sum((proc_time[e][p] * y_vars[e, p].X for p in products if (e, p) in y_vars)) / avail_time[e]) for e in equipments)):.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')