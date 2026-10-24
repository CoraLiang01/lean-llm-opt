CSVQA_DATA = {'query': 'Develop a mathematical optimization model to determine the optimal daily production plan for a wine company that maximizes total net profit. The company produces three distinct wine brands by blending three grades of raw wine material.\n\nThe necessary input data is provided in two files: 30-1.csv contains the daily supply limits and unit costs for the three raw grades, while 30-2.csv provides the selling prices and the blending requirements for all three brands.\n\nThe model should incorporate the following constraints. First, the Blending Requirements constraints specify upper-bound and lower-bound proportion requirements for selected raw grades in each wine brand. Second, the Raw Material Supply constraints require that the total amount of each raw grade used across all brands cannot exceed its corresponding daily supply limit. Third, the Minimum Production constraint requires that the Red brand must have a daily production level of at least 2,000 kg.\n\nThe objective is to determine the quantity of each raw grade allocated to each brand, defined as the decision variables, so as to maximize total net profit, calculated as total sales revenue minus total raw material cost.', 'relationships': [], 'tables': [{'columns': ['Grade', 'Daily Supply (kg)', 'Cost (CNY/kg)'], 'file_name': '30-1.csv', 'records': [{'data_row_number': 0, 'values': {'Cost (CNY/kg)': '6', 'Daily Supply (kg)': '1500', 'Grade': 'I'}}, {'data_row_number': 1, 'values': {'Cost (CNY/kg)': '4.5', 'Daily Supply (kg)': '2000', 'Grade': 'II'}}, {'data_row_number': 2, 'values': {'Cost (CNY/kg)': '3', 'Daily Supply (kg)': '1000', 'Grade': 'III'}}], 'table_id': 'file_0_view_0'}, {'columns': ['Brand', 'Blending Requirements', 'Selling Price (CNY/kg)'], 'file_name': '30-2.csv', 'records': [{'data_row_number': 0, 'values': {'Blending Requirements': 'I less than 10%  II more than 50%', 'Brand': 'Red', 'Selling Price (CNY/kg)': '5.5'}}, {'data_row_number': 1, 'values': {'Blending Requirements': 'III less than 70%  I more than 20%', 'Brand': 'Yellow', 'Selling Price (CNY/kg)': '5'}}, {'data_row_number': 2, 'values': {'Blending Requirements': 'I less than 50%  II more than 10%', 'Brand': 'Blue', 'Selling Price (CNY/kg)': '4.8'}}], 'table_id': 'file_1_view_0'}]}
import gurobipy
from gurobipy import GRB
import re

def solve_problem():
    tables = CSVQA_DATA['tables']
    table0 = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    table1 = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
    G = [rec['values']['Grade'] for rec in table0['records']]
    B = [rec['values']['Brand'] for rec in table1['records']]
    s_g = {}
    c_g = {}
    for rec in table0['records']:
        grade = rec['values']['Grade']
        try:
            s_g[grade] = float(rec['values']['Daily Supply (kg)'])
            c_g[grade] = float(rec['values']['Cost (CNY/kg)'])
        except (KeyError, ValueError) as e:
            raise ValueError(f'Missing or invalid supply/cost for grade {grade}') from e
    p_b = {}
    for rec in table1['records']:
        brand = rec['values']['Brand']
        try:
            p_b[brand] = float(rec['values']['Selling Price (CNY/kg)'])
        except (KeyError, ValueError) as e:
            raise ValueError(f'Missing or invalid price for brand {brand}') from e
    lower_bounds = {}
    upper_bounds = {}
    for rec in table1['records']:
        brand = rec['values']['Brand']
        req_text = rec['values']['Blending Requirements']
        pattern = '([A-Za-z]+)\\s+(less than|more than)\\s+([0-9]+)%'
        for grade, typ, percent in re.findall(pattern, req_text):
            val = int(percent) / 100.0
            key = (brand, grade)
            if typ == 'less than':
                upper_bounds[key] = val
            else:
                lower_bounds[key] = val
    if set(G) != set(s_g.keys()) or set(G) != set(c_g.keys()):
        raise ValueError('Grade set mismatch between supply and cost data')
    if set(B) != set(p_b.keys()):
        raise ValueError('Brand set mismatch between price data')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x_keys = [(g, b) for g in G for b in B]
    x = m.addVars(x_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    P = m.addVars(B, lb=0, vtype=GRB.CONTINUOUS, name='')
    for b in B:
        m.addConstr(P[b] == gurobipy.quicksum((x[g, b] for g in G)), name=f'prod_def_{b}')
    for (b, g), val in lower_bounds.items():
        m.addConstr(x[g, b] >= val * P[b], name=f'lb_{b}_{g}')
    for (b, g), val in upper_bounds.items():
        m.addConstr(x[g, b] <= val * P[b], name=f'ub_{b}_{g}')
    for g in G:
        m.addConstr(gurobipy.quicksum((x[g, b] for b in B)) <= s_g[g], name=f'supply_{g}')
    if 'Red' not in B:
        raise ValueError("Brand 'Red' not found in data")
    m.addConstr(P['Red'] >= 2000, name='min_red')
    revenue = gurobipy.quicksum((p_b[b] * P[b] for b in B))
    cost = gurobipy.quicksum((c_g[g] * gurobipy.quicksum((x[g, b] for b in B)) for g in G))
    m.setObjective(revenue - cost, GRB.MAXIMIZE)
    m.optimize()
    return m
m = solve_problem()
