import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv'
df = pd.read_csv(csv_path, sep=',')
products = ['Product I', 'Product II', 'Product III']
product_short = {'Product I': 'I', 'Product II': 'II', 'Product III': 'III'}
A_equip = ['A1', 'A2']
B_equip = ['B1', 'B2', 'B3']
compat_A = {'Product I': ['A1', 'A2'], 'Product II': ['A1', 'A2'], 'Product III': ['A2']}
compat_B = {'Product I': ['B1', 'B2', 'B3'], 'Product II': ['B1'], 'Product III': ['B2']}

def row_idx(label):
    idx = df['Equipment / Cost'].astype(str).str.strip().str.casefold() == label.strip().casefold()
    matches = df[idx]
    if matches.empty:
        raise KeyError(f"Row '{label}' not found in 'Equipment / Cost'")
    return matches.index[0]
proc_time = {}
for eq in A_equip + B_equip:
    try:
        i = row_idx(eq)
    except KeyError:
        continue
    for p in products:
        val = df.at[i, p]
        if pd.notnull(val):
            proc_time[p, eq] = float(val)
equip_oper_time = {}
equip_full_cost = {}
for eq in A_equip + B_equip:
    try:
        i = row_idx(eq)
    except KeyError:
        continue
    avail_time = df.at[i, 'Available Equipment Operating Time']
    full_cost = df.at[i, 'Equipment Cost at Full Load (yuan)']
    if pd.notnull(avail_time):
        equip_oper_time[eq] = float(avail_time)
    if pd.notnull(full_cost):
        equip_full_cost[eq] = float(full_cost)
rmc_row = row_idx('Raw Material Cost (yuan/unit)')
raw_mat_cost = {}
for p in products:
    val = df.at[rmc_row, p]
    if pd.notnull(val):
        raw_mat_cost[p] = float(val)
price_row = row_idx('Unit Price (yuan/unit)')
unit_price = {}
for p in products:
    val = df.at[price_row, p]
    if pd.notnull(val):
        unit_price[p] = float(val)
model = gp.Model('FactoryProductionPlan')
xA = {}
xB = {}
for p in products:
    for e in compat_A[p]:
        if (p, e) not in proc_time:
            raise ValueError(f'Missing processing time for ({p}, {e}) in procedure A')
        xA[p, e] = model.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name=f'xA_{product_short[p]}_{e}')
    for e in compat_B[p]:
        if (p, e) not in proc_time:
            raise ValueError(f'Missing processing time for ({p}, {e}) in procedure B')
        xB[p, e] = model.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name=f'xB_{product_short[p]}_{e}')
for e in A_equip + B_equip:
    terms = []
    for p, eq in xA:
        if eq == e:
            terms.append(proc_time[p, e] * xA[p, e])
    for p, eq in xB:
        if eq == e:
            terms.append(proc_time[p, e] * xB[p, e])
    if terms:
        if e not in equip_oper_time:
            raise ValueError(f'Missing available operating time for equipment {e}')
        model.addConstr(gp.quicksum(terms) <= equip_oper_time[e], name=f'EquipTime_{e}')
for p in products:
    sumA = gp.quicksum((xA[p, e] for e in compat_A[p]))
    sumB = gp.quicksum((xB[p, e] for e in compat_B[p]))
    model.addConstr(sumA == sumB, name=f'ProdCons_{product_short[p]}')
prod_qty = {}
for p in products:
    prod_qty[p] = gp.quicksum((xA[p, e] for e in compat_A[p]))
total_revenue = gp.quicksum((unit_price[p] * prod_qty[p] for p in products))
total_rm_cost = gp.quicksum((raw_mat_cost[p] * prod_qty[p] for p in products))
equip_cost_terms = []
for e in A_equip + B_equip:
    used_time = gp.LinExpr()
    for p, eq in xA:
        if eq == e:
            used_time += proc_time[p, e] * xA[p, e]
    for p, eq in xB:
        if eq == e:
            used_time += proc_time[p, e] * xB[p, e]
    if e in equip_oper_time and e in equip_full_cost:
        equip_cost_terms.append(used_time / equip_oper_time[e] * equip_full_cost[e])
    elif used_time.getVar(0) is not None and (e not in equip_oper_time or e not in equip_full_cost):
        raise ValueError(f'Missing equipment cost or available time for {e}')
total_equip_cost = gp.quicksum(equip_cost_terms)
model.setObjective(total_revenue - total_rm_cost - total_equip_cost, gp.GRB.MAXIMIZE)
model.optimize()
if model.status == gp.GRB.OPTIMAL:
    print(f'Optimal total profit: {model.objVal:.2f} yuan')
    print('\n--- Production Plan ---')
    for p in products:
        qty = prod_qty[p].getValue()
        print(f'Product {product_short[p]}: {qty:.2f} units')
        print('  Procedure A assignment:')
        for e in compat_A[p]:
            v = xA[p, e].X
            if v > 1e-06:
                print(f'    {e}: {v:.2f} units')
        print('  Procedure B assignment:')
        for e in compat_B[p]:
            v = xB[p, e].X
            if v > 1e-06:
                print(f'    {e}: {v:.2f} units')
    print('\n--- Equipment Utilization ---')
    for e in A_equip + B_equip:
        used = 0.0
        for p, eq in xA:
            if eq == e:
                used += proc_time[p, e] * xA[p, e].X
        for p, eq in xB:
            if eq == e:
                used += proc_time[p, e] * xB[p, e].X
        if e in equip_oper_time:
            print(f'{e}: Used {used:.2f} / {equip_oper_time[e]:.2f} hours ({100 * used / equip_oper_time[e]:.1f}%)')
        else:
            print(f'{e}: Used {used:.2f} hours (no available time data)')
    print('\n--- Cost Breakdown ---')
    print(f'Total revenue: {total_revenue.getValue():.2f} yuan')
    print(f'Total raw material cost: {total_rm_cost.getValue():.2f} yuan')
    print(f'Total equipment cost: {total_equip_cost.getValue():.2f} yuan')
else:
    print(f'No optimal solution found. Status: {model.status}')