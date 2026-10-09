import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv'
df = pd.read_csv(csv_path, sep=',')
products = ['Product I', 'Product II', 'Product III']
product_short = {'Product I': 'I', 'Product II': 'II', 'Product III': 'III'}
procedures = ['A', 'B']
equipment_rows = df['Equipment / Cost'].str.match('^[AB]\\d+$', case=False, na=False)
equipments = df.loc[equipment_rows, 'Equipment / Cost'].tolist()
equip_proc = {}
for eq in equipments:
    if eq.upper().startswith('A'):
        equip_proc[eq] = 'A'
    elif eq.upper().startswith('B'):
        equip_proc[eq] = 'B'
    else:
        raise ValueError(f'Unknown equipment type: {eq}')
eligible_pairs = []
for p in products:
    for e in equipments:
        proc = equip_proc[e]
        if proc == 'A':
            if p == 'Product I' and e in ['A1', 'A2']:
                eligible_pairs.append((p, e))
            elif p == 'Product II' and e in ['A1', 'A2']:
                eligible_pairs.append((p, e))
            elif p == 'Product III' and e == 'A2':
                eligible_pairs.append((p, e))
        elif proc == 'B':
            if p == 'Product I' and e in ['B1', 'B2', 'B3']:
                eligible_pairs.append((p, e))
            elif p == 'Product II' and e == 'B1':
                eligible_pairs.append((p, e))
            elif p == 'Product III' and e == 'B2':
                eligible_pairs.append((p, e))
proc_time = {}
for (idx, row) in df.loc[equipment_rows].iterrows():
    e = row['Equipment / Cost']
    for p in products:
        val = row[p]
        if not (isinstance(val, float) or isinstance(val, int)):
            continue
        if not np.isnan(val):
            proc_time[p, e] = float(val)
for (p, e) in eligible_pairs:
    if (p, e) not in proc_time:
        raise ValueError(f'Missing processing time for ({p}, {e})')
avail_time = {}
equip_cost_full = {}
for (idx, row) in df.loc[equipment_rows].iterrows():
    e = row['Equipment / Cost']
    atime = row['Available Equipment Operating Time']
    ecost = row['Equipment Cost at Full Load (yuan)']
    if not (isinstance(atime, float) or isinstance(atime, int)) or np.isnan(atime):
        raise ValueError(f'Missing available operating time for equipment {e}')
    if not (isinstance(ecost, float) or isinstance(ecost, int)) or np.isnan(ecost):
        raise ValueError(f'Missing equipment cost at full load for equipment {e}')
    avail_time[e] = float(atime)
    equip_cost_full[e] = float(ecost)

def find_param_row(df, key):
    mask = df['Equipment / Cost'].str.strip().str.casefold() == key.casefold()
    if not mask.any():
        raise ValueError(f"Parameter row '{key}' not found in CSV")
    return df.loc[mask].iloc[0]
raw_mat_row = find_param_row(df, 'Raw Material Cost (yuan/unit)')
unit_price_row = find_param_row(df, 'Unit Price (yuan/unit)')
raw_mat_cost = {}
unit_price = {}
for p in products:
    val_rm = raw_mat_row[p]
    val_up = unit_price_row[p]
    if not (isinstance(val_rm, float) or isinstance(val_rm, int)) or np.isnan(val_rm):
        raise ValueError(f'Missing raw material cost for {p}')
    if not (isinstance(val_up, float) or isinstance(val_up, int)) or np.isnan(val_up):
        raise ValueError(f'Missing unit price for {p}')
    raw_mat_cost[p] = float(val_rm)
    unit_price[p] = float(val_up)

def solve_problem():
    m = gp.Model('FactoryProductionPlan')
    x = m.addVars(eligible_pairs, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    prod_A_equip = {p: [e for (pp, e) in eligible_pairs if pp == p and equip_proc[e] == 'A'] for p in products}
    prod_B_equip = {p: [e for (pp, e) in eligible_pairs if pp == p and equip_proc[e] == 'B'] for p in products}
    for p in products:
        m.addConstr(gp.quicksum((x[p, e] for e in prod_A_equip[p])) == gp.quicksum((x[p, e] for e in prod_B_equip[p])), name=f'proc_sync_{product_short[p]}')
    for e in equipments:
        eligible_p = [p for (p, ee) in eligible_pairs if ee == e]
        m.addConstr(gp.quicksum((proc_time[p, e] * x[p, e] for p in eligible_p)) <= avail_time[e], name=f'eq_time_{e}')
    total_prod = {}
    for p in products:
        total_prod[p] = gp.quicksum((x[p, e] for e in prod_A_equip[p]))
    total_revenue = gp.quicksum((unit_price[p] * total_prod[p] for p in products))
    total_raw_mat_cost = gp.quicksum((raw_mat_cost[p] * total_prod[p] for p in products))
    equip_cost = []
    for e in equipments:
        eligible_p = [p for (p, ee) in eligible_pairs if ee == e]
        used_time = gp.quicksum((proc_time[p, e] * x[p, e] for p in eligible_p))
        equip_cost.append(used_time / avail_time[e] * equip_cost_full[e])
    total_equip_cost = gp.quicksum(equip_cost)
    m.setObjective(total_revenue - total_raw_mat_cost - total_equip_cost, gp.GRB.MAXIMIZE)
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal:.6f}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X:.6f}')
else:
    print(f'Solver status: {m.status}')