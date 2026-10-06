import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv'
df = pd.read_csv(csv_path, sep=',')
equipment_rows = df['Equipment / Cost'].str.match('^(A1|A2|B1|B2|B3)$', case=False, na=False)
equipment_df = df.loc[equipment_rows].copy()
product_names = ['Product I', 'Product II', 'Product III']
products = ['I', 'II', 'III']
product_map = dict(zip(product_names, products))
A_equip = ['A1', 'A2']
B_equip = ['B1', 'B2', 'B3']
compat = {('I', 'A'): ['A1', 'A2'], ('I', 'B'): ['B1', 'B2', 'B3'], ('II', 'A'): ['A1', 'A2'], ('II', 'B'): ['B1'], ('III', 'A'): ['A2'], ('III', 'B'): ['B2']}
proc_time = {}
for (_, row) in equipment_df.iterrows():
    eq = row['Equipment / Cost'].strip()
    for (pname, p) in product_map.items():
        val = row[pname]
        if not pd.isnull(val):
            proc_time[p, eq] = float(val)
avail_time = {}
full_load_cost = {}
for (_, row) in equipment_df.iterrows():
    eq = row['Equipment / Cost'].strip()
    atime = row['Available Equipment Operating Time']
    ecost = row['Equipment Cost at Full Load (yuan)']
    if not pd.isnull(atime):
        avail_time[eq] = float(atime)
    if not pd.isnull(ecost):
        full_load_cost[eq] = float(ecost)

def get_param_row(label):
    idx = df['Equipment / Cost'].str.casefold().str.strip() == label.casefold().strip()
    if not idx.any():
        raise ValueError(f"Row '{label}' not found in CSV")
    return df.loc[idx].iloc[0]
raw_mat_row = get_param_row('Raw Material Cost (yuan/unit)')
unit_price_row = get_param_row('Unit Price (yuan/unit)')
raw_mat_cost = {}
unit_price = {}
for (pname, p) in product_map.items():
    val_rm = raw_mat_row[pname]
    val_up = unit_price_row[pname]
    if not pd.isnull(val_rm):
        raw_mat_cost[p] = float(val_rm)
    if not pd.isnull(val_up):
        unit_price[p] = float(val_up)
procs = ['A', 'B']
x_keys = []
for p in products:
    for proc in procs:
        for eq in compat.get((p, proc), []):
            x_keys.append((p, proc, eq))
for (p, proc, eq) in x_keys:
    if (p, eq) not in proc_time:
        raise ValueError(f'Missing processing time for product {p}, equipment {eq}')
for eq in set([k[2] for k in x_keys]):
    if eq not in avail_time or eq not in full_load_cost:
        raise ValueError(f'Missing available time or cost for equipment {eq}')
for p in products:
    if p not in raw_mat_cost or p not in unit_price:
        raise ValueError(f'Missing raw material cost or unit price for product {p}')

def solve_problem():
    m = gp.Model('factory_mixture3')
    m.Params.MIPGap = 0.0001
    q = m.addVars(products, lb=0, vtype=GRB.CONTINUOUS, name='')
    x = m.addVars(x_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    for p in products:
        eqs_A = compat.get((p, 'A'), [])
        m.addConstr(gp.quicksum((x[p, 'A', eq] for eq in eqs_A)) == q[p], name=f'flowA_{p}')
        eqs_B = compat.get((p, 'B'), [])
        m.addConstr(gp.quicksum((x[p, 'B', eq] for eq in eqs_B)) == q[p], name=f'flowB_{p}')
    for eq in set([k[2] for k in x_keys]):
        m.addConstr(gp.quicksum((proc_time[p, eq] * x[p, proc, eq] for (p, proc, eq2) in x_keys if eq2 == eq)) <= avail_time[eq], name=f'eq_time_{eq}')
    eq_usage_time = {}
    for eq in set([k[2] for k in x_keys]):
        eq_usage_time[eq] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f'usage_{eq}')
        m.addConstr(eq_usage_time[eq] == gp.quicksum((proc_time[p, eq] * x[p, proc, eq] for (p, proc, eq2) in x_keys if eq2 == eq)), name=f'usage_def_{eq}')
    revenue = gp.quicksum((unit_price[p] * q[p] for p in products))
    raw_cost = gp.quicksum((raw_mat_cost[p] * q[p] for p in products))
    eq_cost = gp.quicksum((eq_usage_time[eq] / avail_time[eq] * full_load_cost[eq] for eq in eq_usage_time))
    m.setObjective(revenue - raw_cost - eq_cost, GRB.MAXIMIZE)
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')