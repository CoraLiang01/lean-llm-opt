import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv'
    df = pd.read_csv(path, sep=',')
    equipment_rows = df[df['Available Equipment Operating Time'].notnull()].copy()
    equipment_rows['Equipment / Cost'] = equipment_rows['Equipment / Cost'].astype(str).str.strip()
    raw_material_row = df[df['Equipment / Cost'].str.strip().str.casefold() == 'raw material cost (yuan/unit)']
    unit_price_row = df[df['Equipment / Cost'].str.strip().str.casefold() == 'unit price (yuan/unit)']
    products = ['Product I', 'Product II', 'Product III']
    product_short = {'Product I': 'I', 'Product II': 'II', 'Product III': 'III'}
    procedures = ['A', 'B']
    procA_equips = ['A1', 'A2']
    procB_equips = ['B1', 'B2', 'B3']
    all_equips = procA_equips + procB_equips
    eligibility = {('A', 'Product I'): ['A1', 'A2'], ('A', 'Product II'): ['A1', 'A2'], ('A', 'Product III'): ['A2'], ('B', 'Product I'): ['B1', 'B2', 'B3'], ('B', 'Product II'): ['B1'], ('B', 'Product III'): ['B2']}
    proc_time = {}
    for proc, equips in [('A', procA_equips), ('B', procB_equips)]:
        for e in equips:
            row = equipment_rows[equipment_rows['Equipment / Cost'] == e]
            if row.empty:
                raise ValueError(f'Equipment {e} not found in CSV.')
            for p in products:
                if (proc, p) in eligibility and e in eligibility[proc, p]:
                    val = row.iloc[0][p]
                    if pd.isnull(val):
                        raise ValueError(f'Missing processing time for equipment {e}, product {p}.')
                    proc_time[proc, e, p] = float(val)
    raw_material_cost = {}
    unit_price = {}
    for p in products:
        val = raw_material_row[p].values
        if len(val) == 0 or pd.isnull(val[0]):
            raise ValueError(f'Missing raw material cost for {p}.')
        raw_material_cost[p] = float(val[0])
        val = unit_price_row[p].values
        if len(val) == 0 or pd.isnull(val[0]):
            raise ValueError(f'Missing unit price for {p}.')
        unit_price[p] = float(val[0])
    equip_oper_time = {}
    equip_full_cost = {}
    for e in all_equips:
        row = equipment_rows[equipment_rows['Equipment / Cost'] == e]
        if row.empty:
            raise ValueError(f'Equipment {e} not found in CSV.')
        oper_time = row.iloc[0]['Available Equipment Operating Time']
        full_cost = row.iloc[0]['Equipment Cost at Full Load (yuan)']
        if pd.isnull(oper_time) or pd.isnull(full_cost):
            raise ValueError(f'Missing operating time or cost for equipment {e}.')
        equip_oper_time[e] = float(oper_time)
        equip_full_cost[e] = float(full_cost)
    m = gp.Model('factory_mixture3')
    q = m.addVars(products, lb=0, vtype=GRB.CONTINUOUS, name='')
    x_keys = []
    for proc in procedures:
        equips = procA_equips if proc == 'A' else procB_equips
        for p in products:
            if (proc, p) in eligibility:
                for e in eligibility[proc, p]:
                    x_keys.append((proc, e, p))
    x = m.addVars(x_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    for p in products:
        if ('A', p) in eligibility:
            m.addConstr(gp.quicksum((x['A', e, p] for e in eligibility['A', p])) == q[p], name=f'assign_A_{product_short[p]}')
        if ('B', p) in eligibility:
            m.addConstr(gp.quicksum((x['B', e, p] for e in eligibility['B', p])) == q[p], name=f'assign_B_{product_short[p]}')
    for e in all_equips:
        terms = []
        if e in procA_equips:
            for p in products:
                if ('A', p) in eligibility and e in eligibility['A', p]:
                    terms.append(proc_time['A', e, p] * x['A', e, p])
        if e in procB_equips:
            for p in products:
                if ('B', p) in eligibility and e in eligibility['B', p]:
                    terms.append(proc_time['B', e, p] * x['B', e, p])
        m.addConstr(gp.quicksum(terms) <= equip_oper_time[e], name=f'cap_{e}')
    revenue = gp.quicksum((unit_price[p] * q[p] for p in products))
    raw_cost = gp.quicksum((raw_material_cost[p] * q[p] for p in products))
    equip_cost_terms = []
    for e in all_equips:
        used_time = []
        if e in procA_equips:
            for p in products:
                if ('A', p) in eligibility and e in eligibility['A', p]:
                    used_time.append(proc_time['A', e, p] * x['A', e, p])
        if e in procB_equips:
            for p in products:
                if ('B', p) in eligibility and e in eligibility['B', p]:
                    used_time.append(proc_time['B', e, p] * x['B', e, p])
        total_used_time = gp.quicksum(used_time)
        equip_cost_terms.append(equip_full_cost[e] * (total_used_time / equip_oper_time[e]))
    equip_cost = gp.quicksum(equip_cost_terms)
    m.setObjective(revenue - raw_cost - equip_cost, GRB.MAXIMIZE)
    m.optimize()
    return m
m = solve_problem()