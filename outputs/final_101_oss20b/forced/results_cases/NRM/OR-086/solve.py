CSVQA_DATA = {'query': 'Develop a mathematical optimization model to determine the optimal daily production plan for a wine company that maximizes total net profit. The company produces three distinct wine brands by blending three grades of raw wine material.\n\nThe necessary input data is provided in two files: 30-1.csv contains the daily supply limits and unit costs for the three raw grades, while 30-2.csv provides the selling prices and the blending requirements for all three brands.\n\nThe model should incorporate the following constraints. First, the Blending Requirements constraints specify upper-bound and lower-bound proportion requirements for selected raw grades in each wine brand. Second, the Raw Material Supply constraints require that the total amount of each raw grade used across all brands cannot exceed its corresponding daily supply limit. Third, the Minimum Production constraint requires that the Red brand must have a daily production level of at least 2,000 kg.\n\nThe objective is to determine the quantity of each raw grade allocated to each brand, defined as the decision variables, so as to maximize total net profit, calculated as total sales revenue minus total raw material cost.', 'relationships': [], 'tables': [{'columns': ['Grade', 'Daily Supply (kg)', 'Cost (CNY/kg)'], 'file_name': '30-1.csv', 'records': [{'data_row_number': 0, 'values': {'Cost (CNY/kg)': '6', 'Daily Supply (kg)': '1500', 'Grade': 'I'}}, {'data_row_number': 1, 'values': {'Cost (CNY/kg)': '4.5', 'Daily Supply (kg)': '2000', 'Grade': 'II'}}, {'data_row_number': 2, 'values': {'Cost (CNY/kg)': '3', 'Daily Supply (kg)': '1000', 'Grade': 'III'}}], 'table_id': 'file_0_view_0'}, {'columns': ['Brand', 'Blending Requirements', 'Selling Price (CNY/kg)'], 'file_name': '30-2.csv', 'records': [{'data_row_number': 0, 'values': {'Blending Requirements': 'I less than 10%  II more than 50%', 'Brand': 'Red', 'Selling Price (CNY/kg)': '5.5'}}, {'data_row_number': 1, 'values': {'Blending Requirements': 'III less than 70%  I more than 20%', 'Brand': 'Yellow', 'Selling Price (CNY/kg)': '5'}}, {'data_row_number': 2, 'values': {'Blending Requirements': 'I less than 50%  II more than 10%', 'Brand': 'Blue', 'Selling Price (CNY/kg)': '4.8'}}], 'table_id': 'file_1_view_0'}]}
import gurobipy
from gurobipy import GRB
import re
tables = CSVQA_DATA['tables']
table0 = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
table1 = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
G = []
c_g = {}
s_g = {}
for rec in table0['records']:
    grade = rec['values']['Grade']
    G.append(grade)
    c_g[grade] = float(rec['values']['Cost (CNY/kg)'])
    s_g[grade] = float(rec['values']['Daily Supply (kg)'])
B = []
p_b = {}
L_b_g = {}
U_b_g = {}
for rec in table1['records']:
    brand = rec['values']['Brand']
    B.append(brand)
    p_b[brand] = float(rec['values']['Selling Price (CNY/kg)'])
    req_text = rec['values']['Blending Requirements']
    clauses = re.split('\\s{2,}', req_text.strip())
    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue
        grade = clause.split()[0]
        if 'less than' in clause:
            m = re.search('(\\d+(\\.\\d+)?)%', clause)
            if m:
                val = float(m.group(1)) / 100
                U_b_g[brand, grade] = min(U_b_g.get((brand, grade), 1.0), val)
        elif 'more than' in clause:
            m = re.search('(\\d+(\\.\\d+)?)%', clause)
            if m:
                val = float(m.group(1)) / 100
                L_b_g[brand, grade] = max(L_b_g.get((brand, grade), 0.0), val)
for b in B:
    for g in G:
        if (b, g) not in L_b_g:
            L_b_g[b, g] = 0.0
        if (b, g) not in U_b_g:
            U_b_g[b, g] = 1.0
for g in G:
    if g not in c_g or g not in s_g:
        raise ValueError(f'Missing cost or supply for grade {g}')
for b in B:
    if b not in p_b:
        raise ValueError(f'Missing price for brand {b}')

def solve_problem():
    m = gurobipy.Model('wine_blending')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(g, b) for g in G for b in B], lb=0, vtype=GRB.CONTINUOUS, name='')
    T = m.addVars(B, lb=0, vtype=GRB.CONTINUOUS, name='')
    obj = gurobipy.quicksum((p_b[b] * T[b] for b in B)) - gurobipy.quicksum((c_g[g] * x[g, b] for g in G for b in B))
    m.setObjective(obj, GRB.MAXIMIZE)
    for b in B:
        m.addConstr(T[b] == gurobipy.quicksum((x[g, b] for g in G)), name=f'prod_def_{b}')
    for b in B:
        for g in G:
            L = L_b_g[b, g]
            U = U_b_g[b, g]
            m.addConstr(x[g, b] >= L * T[b], name=f'lb_{b}_{g}')
            m.addConstr(x[g, b] <= U * T[b], name=f'ub_{b}_{g}')
    for g in G:
        m.addConstr(gurobipy.quicksum((x[g, b] for b in B)) <= s_g[g], name=f'supply_{g}')
    if 'Red' in B:
        m.addConstr(T['Red'] >= 2000, name='min_red')
    else:
        raise ValueError("Brand 'Red' not found in data")
    m.optimize()
    return m
m = solve_problem()
