CSVQA_DATA = {'query': 'Develop a mathematical optimization model to determine the optimal daily production plan for a wine company that maximizes total net profit. The company produces three distinct wine brands by blending three grades of raw wine material.\n\nThe necessary input data is provided in two files: 30-1.csv contains the daily supply limits and unit costs for the three raw grades, while 30-2.csv provides the selling prices and the blending requirements for all three brands.\n\nThe model should incorporate the following constraints. First, the Blending Requirements constraints specify upper-bound and lower-bound proportion requirements for selected raw grades in each wine brand. Second, the Raw Material Supply constraints require that the total amount of each raw grade used across all brands cannot exceed its corresponding daily supply limit. Third, the Minimum Production constraint requires that the Red brand must have a daily production level of at least 2,000 kg.\n\nThe objective is to determine the quantity of each raw grade allocated to each brand, defined as the decision variables, so as to maximize total net profit, calculated as total sales revenue minus total raw material cost.', 'relationships': [], 'tables': [{'columns': ['Grade', 'Daily Supply (kg)', 'Cost (CNY/kg)'], 'file_name': '30-1.csv', 'records': [{'data_row_number': 0, 'values': {'Cost (CNY/kg)': '6', 'Daily Supply (kg)': '1500', 'Grade': 'I'}}, {'data_row_number': 1, 'values': {'Cost (CNY/kg)': '4.5', 'Daily Supply (kg)': '2000', 'Grade': 'II'}}, {'data_row_number': 2, 'values': {'Cost (CNY/kg)': '3', 'Daily Supply (kg)': '1000', 'Grade': 'III'}}], 'table_id': 'file_0_view_0'}, {'columns': ['Brand', 'Selling Price (CNY/kg)', 'Blending Requirements'], 'file_name': '30-2.csv', 'records': [{'data_row_number': 0, 'values': {'Blending Requirements': 'I less than 10%  II more than 50%', 'Brand': 'Red', 'Selling Price (CNY/kg)': '5.5'}}, {'data_row_number': 1, 'values': {'Blending Requirements': 'III less than 70%  I more than 20%', 'Brand': 'Yellow', 'Selling Price (CNY/kg)': '5'}}, {'data_row_number': 2, 'values': {'Blending Requirements': 'I less than 50%  II more than 10%', 'Brand': 'Blue', 'Selling Price (CNY/kg)': '4.8'}}], 'table_id': 'file_1_view_0'}]}
import gurobipy
from gurobipy import GRB
import re

def solve_problem():
    data = CSVQA_DATA
    table0 = next((t for t in data['tables'] if t['table_id'] == 'file_0_view_0'))
    table1 = next((t for t in data['tables'] if t['table_id'] == 'file_1_view_0'))
    G = [rec['values']['Grade'] for rec in table0['records']]
    B = [rec['values']['Brand'] for rec in table1['records']]
    if not G:
        raise ValueError('No grades found in table 0')
    if not B:
        raise ValueError('No brands found in table 1')
    s_g = {}
    c_g = {}
    for rec in table0['records']:
        grade = rec['values']['Grade']
        if grade in s_g or grade in c_g:
            raise ValueError(f'Duplicate grade entry: {grade}')
        s_g[grade] = float(rec['values']['Daily Supply (kg)'])
        c_g[grade] = float(rec['values']['Cost (CNY/kg)'])
    for g in G:
        if g not in s_g:
            raise ValueError(f'Missing supply for grade {g}')
        if g not in c_g:
            raise ValueError(f'Missing cost for grade {g}')
    p_b = {}
    for rec in table1['records']:
        brand = rec['values']['Brand']
        if brand in p_b:
            raise ValueError(f'Duplicate brand entry: {brand}')
        p_b[brand] = float(rec['values']['Selling Price (CNY/kg)'])
    for b in B:
        if b not in p_b:
            raise ValueError(f'Missing price for brand {b}')
    lower = {}
    upper = {}
    pattern = re.compile('([A-Za-z]+)\\s+(less than|more than)\\s+(\\d+)%')
    for rec in table1['records']:
        brand = rec['values']['Brand']
        req = rec['values']['Blending Requirements']
        for match in pattern.finditer(req):
            grade, typ, pct = match.groups()
            pct_val = int(pct) / 100.0
            if typ == 'more than':
                lower[brand, grade] = pct_val
            else:
                upper[brand, grade] = pct_val
    for b in B:
        for g in G:
            if (b, g) not in lower:
                lower[b, g] = 0.0
            if (b, g) not in upper:
                upper[b, g] = 1.0
    m = gurobipy.Model('wine_blending')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(b, g) for b in B for g in G], lb=0, vtype=GRB.CONTINUOUS, name='')
    P = m.addVars(B, lb=0, vtype=GRB.CONTINUOUS, name='')
    obj = gurobipy.quicksum((p_b[b] * P[b] for b in B)) - gurobipy.quicksum((c_g[g] * x[b, g] for b in B for g in G))
    m.setObjective(obj, GRB.MAXIMIZE)
    m.addConstrs((P[b] == gurobipy.quicksum((x[b, g] for g in G)) for b in B), name='')
    m.addConstrs((x[b, g] >= lower[b, g] * P[b] for b, g in lower if lower[b, g] > 0), name='')
    m.addConstrs((x[b, g] <= upper[b, g] * P[b] for b, g in upper if upper[b, g] < 1), name='')
    m.addConstrs((gurobipy.quicksum((x[b, g] for b in B)) <= s_g[g] for g in G), name='')
    if 'Red' not in B:
        raise ValueError("Brand 'Red' not found in data")
    m.addConstr(P['Red'] >= 2000, name='min_red')
    m.optimize()
    return m
m = solve_problem()
