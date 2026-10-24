CSVQA_DATA = {'query': 'Develop a mathematical optimization model to determine the optimal daily production plan for a wine company that maximizes total net profit. The company produces three distinct wine brands by blending three grades of raw wine material.\n\nThe necessary input data is provided in two files: 30-1.csv contains the daily supply limits and unit costs for the three raw grades, while 30-2.csv provides the selling prices and the blending requirements for all three brands.\n\nThe model should incorporate the following constraints. First, the Blending Requirements constraints specify upper-bound and lower-bound proportion requirements for selected raw grades in each wine brand. Second, the Raw Material Supply constraints require that the total amount of each raw grade used across all brands cannot exceed its corresponding daily supply limit. Third, the Minimum Production constraint requires that the Red brand must have a daily production level of at least 2,000 kg.\n\nThe objective is to determine the quantity of each raw grade allocated to each brand, defined as the decision variables, so as to maximize total net profit, calculated as total sales revenue minus total raw material cost.', 'relationships': [], 'tables': [{'columns': ['Grade', 'Daily Supply (kg)', 'Cost (CNY/kg)'], 'file_name': '30-1.csv', 'records': [{'data_row_number': 0, 'values': {'Cost (CNY/kg)': '6', 'Daily Supply (kg)': '1500', 'Grade': 'I'}}, {'data_row_number': 1, 'values': {'Cost (CNY/kg)': '4.5', 'Daily Supply (kg)': '2000', 'Grade': 'II'}}, {'data_row_number': 2, 'values': {'Cost (CNY/kg)': '3', 'Daily Supply (kg)': '1000', 'Grade': 'III'}}], 'table_id': 'file_0_view_0'}, {'columns': ['Brand', 'Blending Requirements', 'Selling Price (CNY/kg)'], 'file_name': '30-2.csv', 'records': [{'data_row_number': 0, 'values': {'Blending Requirements': 'I less than 10%  II more than 50%', 'Brand': 'Red', 'Selling Price (CNY/kg)': '5.5'}}, {'data_row_number': 1, 'values': {'Blending Requirements': 'III less than 70%  I more than 20%', 'Brand': 'Yellow', 'Selling Price (CNY/kg)': '5'}}, {'data_row_number': 2, 'values': {'Blending Requirements': 'I less than 50%  II more than 10%', 'Brand': 'Blue', 'Selling Price (CNY/kg)': '4.8'}}], 'table_id': 'file_1_view_0'}]}
import gurobipy
from gurobipy import GRB
import re
data = CSVQA_DATA

def get_table(table_id):
    for tbl in data['tables']:
        if tbl['table_id'] == table_id:
            return tbl
    raise ValueError(f'Table {table_id} not found')
table0 = get_table('file_0_view_0')
grades = []
S_g = {}
c_g = {}
for rec in table0['records']:
    grade = rec['values']['Grade'].strip()
    grades.append(grade)
    S_g[grade] = float(rec['values']['Daily Supply (kg)'])
    c_g[grade] = float(rec['values']['Cost (CNY/kg)'])
table1 = get_table('file_1_view_0')
brands = []
p_b = {}
l_gb = {}
u_gb = {}
for g in grades:
    for b in []:
        pass
for rec in table1['records']:
    brand = rec['values']['Brand'].strip()
    brands.append(brand)
    p_b[brand] = float(rec['values']['Selling Price (CNY/kg)'])
    req_str = rec['values']['Blending Requirements']
    for g in grades:
        l_gb[g, brand] = 0.0
        u_gb[g, brand] = 1.0
    pattern = '(?P<grade>I|II|III)\\s+(?P<type>less|more)\\s+than\\s+(?P<percent>\\d+)%'
    for m in re.finditer(pattern, req_str):
        g = m.group('grade')
        typ = m.group('type')
        pct = float(m.group('percent')) / 100.0
        if typ == 'less':
            u_gb[g, brand] = pct
        else:
            l_gb[g, brand] = pct
for g in grades:
    if g not in S_g:
        raise ValueError(f'Missing supply limit for grade {g}')
    if g not in c_g:
        raise ValueError(f'Missing cost for grade {g}')
for b in brands:
    if b not in p_b:
        raise ValueError(f'Missing selling price for brand {b}')
for g in grades:
    for b in brands:
        if (g, b) not in l_gb or (g, b) not in u_gb:
            raise ValueError(f'Missing bounds for grade {g} and brand {b}')

def solve_problem():
    m = gurobipy.Model('wine_blending')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(g, b) for g in grades for b in brands], lb=0, vtype=GRB.CONTINUOUS, name='')
    revenue = gurobipy.quicksum((p_b[b] * gurobipy.quicksum((x[g, b] for g in grades)) for b in brands))
    cost = gurobipy.quicksum((c_g[g] * gurobipy.quicksum((x[g, b] for b in brands)) for g in grades))
    m.setObjective(revenue - cost, GRB.MAXIMIZE)
    for g in grades:
        for b in brands:
            total_b = gurobipy.quicksum((x[h, b] for h in grades))
            m.addConstr(l_gb[g, b] * total_b <= x[g, b], name=f'lb_{g}_{b}')
            m.addConstr(x[g, b] <= u_gb[g, b] * total_b, name=f'ub_{g}_{b}')
    for g in grades:
        m.addConstr(gurobipy.quicksum((x[g, b] for b in brands)) <= S_g[g], name=f'supply_{g}')
    m.addConstr(gurobipy.quicksum((x[g, 'Red'] for g in grades)) >= 2000, name='min_red')
    m.optimize()
    return m
m = solve_problem()
