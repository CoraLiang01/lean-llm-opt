import gurobipy as gp
import pandas as pd
import numpy as np
import re

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv'
    df = pd.read_csv(csv_path, sep=',')
    df.columns = [col.strip() for col in df.columns]
    products = ['Product I', 'Product II', 'Product III']
    procs = {'A': ['A1', 'A2'], 'B': ['B1', 'B2', 'B3']}
    all_equipment = procs['A'] + procs['B']
    prod_short = {'I': 'Product I', 'II': 'Product II', 'III': 'Product III'}
    prod_full = {v: k for (k, v) in prod_short.items()}
    df['Equipment / Cost'] = df['Equipment / Cost'].str.strip()
    selling_price_row = df['Equipment / Cost'].str.casefold() == 'selling price (yuan/unit)'.casefold()
    raw_material_cost_row = df['Equipment / Cost'].str.casefold() == 'raw material cost (yuan/unit)'.casefold()
    if not selling_price_row.any() or not raw_material_cost_row.any():
        raise ValueError('Missing selling price or raw material cost row in CSV.')
    selling_price = {}
    raw_material_cost = {}
    for p in products:
        selling_price[p] = float(df.loc[selling_price_row, p].values[0])
        raw_material_cost[p] = float(df.loc[raw_material_cost_row, p].values[0])
    processing_time = {}
    for e in all_equipment:
        eq_row = df['Equipment / Cost'].str.casefold() == e.casefold()
        if not eq_row.any():
            raise ValueError(f'Missing equipment row for {e} in CSV.')
        for p in products:
            val = df.loc[eq_row, p].values[0]
            if pd.isnull(val):
                continue
            try:
                processing_time[p, e] = float(val)
            except Exception:
                raise ValueError(f'Invalid processing time for ({p}, {e}): {val}')
    available_time = {}
    full_load_cost = {}
    for e in all_equipment:
        eq_row = df['Equipment / Cost'].str.casefold() == e.casefold()
        if not eq_row.any():
            raise ValueError(f'Missing equipment row for {e} in CSV.')
        atime = df.loc[eq_row, 'Available Equipment Operating Time'].values[0]
        acost = df.loc[eq_row, 'Equipment Cost at Full Load (yuan)'].values[0]
        if pd.isnull(atime) or pd.isnull(acost):
            raise ValueError(f'Missing available time or cost for equipment {e}.')
        available_time[e] = float(atime)
        full_load_cost[e] = float(acost)
    allowed_z = []
    for e in procs['A']:
        allowed_z.append(('Product I', e))
    for e in procs['A']:
        allowed_z.append(('Product II', e))
    allowed_z.append(('Product III', 'A2'))
    for e in procs['B']:
        allowed_z.append(('Product I', e))
    allowed_z.append(('Product II', 'B1'))
    allowed_z.append(('Product III', 'B2'))
    for (p, e) in allowed_z:
        if (p, e) not in processing_time:
            raise ValueError(f'Missing processing time for allowed pair ({p}, {e})')
    m = gp.Model('FactoryProduction')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(products, lb=0, vtype=gp.GRB.CONTINUOUS, name='')
    z = m.addVars(allowed_z, lb=0, vtype=gp.GRB.CONTINUOUS, name='')
    m.addConstr(z['Product I', 'A1'] + z['Product I', 'A2'] == x['Product I'], name='assign_I_A')
    m.addConstr(z['Product II', 'A1'] + z['Product II', 'A2'] == x['Product II'], name='assign_II_A')
    m.addConstr(z['Product III', 'A2'] == x['Product III'], name='assign_III_A')
    m.addConstr(z['Product I', 'B1'] + z['Product I', 'B2'] + z['Product I', 'B3'] == x['Product I'], name='assign_I_B')
    m.addConstr(z['Product II', 'B1'] == x['Product II'], name='assign_II_B')
    m.addConstr(z['Product III', 'B2'] == x['Product III'], name='assign_III_B')
    for e in all_equipment:
        z_pe = []
        for (p, eq) in allowed_z:
            if eq == e:
                z_pe.append((p, eq))
        expr = gp.quicksum((z[p, eq] * processing_time[p, eq] for (p, eq) in z_pe))
        m.addConstr(expr <= available_time[e], name=f'cap_{e}')
    total_revenue = gp.quicksum((selling_price[p] * x[p] for p in products))
    total_raw_cost = gp.quicksum((raw_material_cost[p] * x[p] for p in products))
    equipment_cost_terms = []
    for e in all_equipment:
        z_pe = []
        for (p, eq) in allowed_z:
            if eq == e:
                z_pe.append((p, eq))
        total_time_used = gp.quicksum((z[p, eq] * processing_time[p, eq] for (p, eq) in z_pe))
        if available_time[e] == 0:
            raise ValueError(f'Available time for equipment {e} is zero.')
        equipment_cost_terms.append(total_time_used / available_time[e] * full_load_cost[e])
    total_equipment_cost = gp.quicksum(equipment_cost_terms)
    m.setObjective(total_revenue - total_raw_cost - total_equipment_cost, gp.GRB.MAXIMIZE)
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal:.6f}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X:.6f}')
else:
    print(f'Solver status: {m.status}')