import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import re

def solve_problem():
    path_30_1 = 'Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv'
    path_30_2 = 'Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv'
    df1 = pd.read_csv(path_30_1, dtype=str, keep_default_na=False)
    S_g = {}
    C_g = {}
    G = []
    for (_, row) in df1.iterrows():
        grade = row['Grade'].strip()
        if grade == '':
            continue
        G.append(grade)
        try:
            S_g[grade] = float(row['Daily Supply (kg)'])
            C_g[grade] = float(row['Cost (CNY/kg)'])
        except ValueError as e:
            raise ValueError(f'Invalid numeric value for grade {grade}: {e}')
    df2 = pd.read_csv(path_30_2, dtype=str, keep_default_na=False)
    P_b = {}
    B = []
    L = {}
    U = {}
    for g in G:
        for b in []:
            pass
    for (_, row) in df2.iterrows():
        brand = row['Brand'].strip()
        if brand == '':
            continue
        B.append(brand)
        try:
            P_b[brand] = float(row['Selling Price (CNY/kg)'])
        except ValueError as e:
            raise ValueError(f'Invalid selling price for brand {brand}: {e}')
    for g in G:
        for b in B:
            L[g, b] = 0.0
            U[g, b] = 1.0
    pattern = re.compile('([A-Za-z]+)\\s+(less than|more than)\\s+(\\d+)%')
    for (_, row) in df2.iterrows():
        brand = row['Brand'].strip()
        req_str = row['Blending Requirements'].strip()
        for match in pattern.finditer(req_str):
            grade = match.group(1).strip()
            relation = match.group(2).strip()
            percent = int(match.group(3))
            if (grade, brand) not in L:
                raise ValueError(f'Unknown grade {grade} in blending requirements for brand {brand}')
            if relation == 'less than':
                U[grade, brand] = percent / 100.0
            elif relation == 'more than':
                L[grade, brand] = percent / 100.0
            else:
                raise ValueError(f'Unknown relation {relation} in blending requirements')
    for g in G:
        if g not in S_g or g not in C_g:
            raise ValueError(f'Missing supply or cost data for grade {g}')
    for b in B:
        if b not in P_b:
            raise ValueError(f'Missing selling price data for brand {b}')
    m = gp.Model('WineBlending')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(g, b) for g in G for b in B], lb=0, vtype=GRB.CONTINUOUS, name='')
    for g in G:
        m.addConstr(gp.quicksum((x[g, b] for b in B)) <= S_g[g], name=f'supply_{g}')
    if 'Red' not in B:
        raise ValueError("Brand 'Red' not found in data")
    m.addConstr(gp.quicksum((x[g, 'Red'] for g in G)) >= 2000, name='min_red')
    for g in G:
        for b in B:
            Q_b = gp.quicksum((x[h, b] for h in G))
            m.addConstr(L[g, b] * Q_b <= x[g, b], name=f'lb_{g}_{b}')
            m.addConstr(x[g, b] <= U[g, b] * Q_b, name=f'ub_{g}_{b}')
    m.setObjective(gp.quicksum(((P_b[b] - C_g[g]) * x[g, b] for g in G for b in B)), GRB.MAXIMIZE)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'Optimal objective value: {m.ObjVal:.4f}')
        for var in m.getVars():
            print(f'{var.VarName} = {var.X:.4f}')
    return m
m = solve_problem()