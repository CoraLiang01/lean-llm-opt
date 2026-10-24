import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    path = 'Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv'
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    P = ['I', 'II', 'III']
    E = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'B4']
    prod_cols = ['Product I', 'Product II', 'Product III']
    avail_col = 'Available Equipment Operating Time'
    cost_col = 'Equipment Cost at Full Load (yuan)'
    raw_cost_row = 'Raw Material Cost (yuan/unit)'
    unit_price_row = 'Unit Price (yuan/unit)'

    def to_float(s):
        return float(s) if s != '' else None
    t = {p: {} for p in P}
    a_e = {}
    c_e = {}
    p_p = {}
    r_p = {}
    for (_, row) in df.iterrows():
        equip = row['Equipment / Cost'].strip()
        if equip == raw_cost_row:
            for (p, col) in zip(P, prod_cols):
                val = to_float(row[col])
                if val is None:
                    raise ValueError(f'Missing raw material cost for product {p}')
                r_p[p] = val
        elif equip == unit_price_row:
            for (p, col) in zip(P, prod_cols):
                val = to_float(row[col])
                if val is None:
                    raise ValueError(f'Missing unit price for product {p}')
                p_p[p] = val
        else:
            if equip not in E:
                continue
            a_val = to_float(row[avail_col])
            if a_val is None:
                raise ValueError(f'Missing available time for equipment {equip}')
            a_e[equip] = a_val
            c_val = to_float(row[cost_col])
            if c_val is None:
                raise ValueError(f'Missing equipment cost for equipment {equip}')
            c_e[equip] = c_val
            for (p, col) in zip(P, prod_cols):
                val = to_float(row[col])
                if val is not None:
                    t[p][equip] = val
    for p in P:
        if p not in p_p:
            raise ValueError(f'Missing unit price for product {p}')
        if p not in r_p:
            raise ValueError(f'Missing raw material cost for product {p}')
    for e in E:
        if e not in a_e:
            raise ValueError(f'Missing available time for equipment {e}')
        if e not in c_e:
            raise ValueError(f'Missing equipment cost for equipment {e}')
    keys = [(p, e) for p in P for e in E if e in t[p]]
    m = gurobipy.Model('ProductionProfit')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    obj_expr = gurobipy.quicksum((x[p, e] * (p_p[p] - r_p[p] - c_e[e] * t[p][e]) for (p, e) in keys))
    m.setObjective(obj_expr, GRB.MAXIMIZE)
    for e in E:
        expr = gurobipy.quicksum((t[p][e] * x[p, e] for p in P if e in t[p]))
        m.addConstr(expr <= a_e[e], name=f'Cap_{e}')
    m.optimize()
    if m.status == GRB.OPTIMAL:
        print(f'Optimal objective: {m.objVal:.6f}')
        for var in m.getVars():
            print(f'{var.VarName} {var.X:.6f}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_problem()