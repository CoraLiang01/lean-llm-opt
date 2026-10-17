CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'Develop a mathematical optimization model to determine the optimal daily production plan for a wine company '
          'that maximizes total net profit. The company produces three distinct wine brands by blending three grades '
          'of raw wine material.\n'
          '\n'
          'The necessary input data is provided in two files: 30-1.csv contains the daily supply limits and unit costs '
          'for the three raw grades, while 30-2.csv provides the selling prices and the blending requirements for all '
          'three brands.\n'
          '\n'
          'The model should incorporate the following constraints. First, the Blending Requirements constraints '
          'specify upper-bound and lower-bound proportion requirements for selected raw grades in each wine brand. '
          'Second, the Raw Material Supply constraints require that the total amount of each raw grade used across all '
          'brands cannot exceed its corresponding daily supply limit. Third, the Minimum Production constraint '
          'requires that the Red brand must have a daily production level of at least 2,000 kg.\n'
          '\n'
          'The objective is to determine the quantity of each raw grade allocated to each brand, defined as the '
          'decision variables, so as to maximize total net profit, calculated as total sales revenue minus total raw '
          'material cost.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Grade', 'Daily Supply (kg)', 'Cost (CNY/kg)'],
             'file_index': 0,
             'file_name': '30-1.csv',
             'filters': {},
             'original_rows': 3,
             'records': [{'source_row': 0, 'values': {'Cost (CNY/kg)': '6', 'Daily Supply (kg)': '1500', 'Grade': 'I'}},
                         {'source_row': 1,
                          'values': {'Cost (CNY/kg)': '4.5', 'Daily Supply (kg)': '2000', 'Grade': 'II'}},
                         {'source_row': 2,
                          'values': {'Cost (CNY/kg)': '3', 'Daily Supply (kg)': '1000', 'Grade': 'III'}}],
             'returned_rows': 3,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'},
            {'columns': ['Brand', 'Blending Requirements', 'Selling Price (CNY/kg)'],
             'file_index': 1,
             'file_name': '30-2.csv',
             'filters': {},
             'original_rows': 3,
             'records': [{'source_row': 0,
                          'values': {'Blending Requirements': 'I less than 10%  II more than 50%',
                                     'Brand': 'Red',
                                     'Selling Price (CNY/kg)': '5.5'}},
                         {'source_row': 1,
                          'values': {'Blending Requirements': 'III less than 70%  I more than 20%',
                                     'Brand': 'Yellow',
                                     'Selling Price (CNY/kg)': '5'}},
                         {'source_row': 2,
                          'values': {'Blending Requirements': 'I less than 50%  II more than 10%',
                                     'Brand': 'Blue',
                                     'Selling Price (CNY/kg)': '4.8'}}],
             'returned_rows': 3,
             'role': 'descriptive non-unique role',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB
import re

def parse_requirements(req_str):
    lower = {}
    upper = {}
    pattern = '([A-Za-z]+)\\s+(less than|more than)\\s+(\\d+)%'
    for grade, relation, percent in re.findall(pattern, req_str):
        value = float(percent) / 100.0
        if relation == 'more than':
            lower[grade] = value
        elif relation == 'less than':
            upper[grade] = value
    return (lower, upper)

def solve_problem():
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    tables = CSVQA_DATA['tables']
    table0 = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    supply = {}
    cost = {}
    grades = set()
    for rec in table0['records']:
        g = rec['values']['Grade'].strip()
        grades.add(g)
        supply[g] = float(rec['values']['Daily Supply (kg)'])
        cost[g] = float(rec['values']['Cost (CNY/kg)'])
    table1 = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
    price = {}
    lower_bounds = {}
    upper_bounds = {}
    brands = set()
    for rec in table1['records']:
        b = rec['values']['Brand'].strip()
        brands.add(b)
        price[b] = float(rec['values']['Selling Price (CNY/kg)'])
        req_str = rec['values']['Blending Requirements']
        lower, upper = parse_requirements(req_str)
        for g, val in lower.items():
            lower_bounds[b, g] = val
        for g, val in upper.items():
            upper_bounds[b, g] = val
    for g in grades:
        if g not in supply:
            raise ValueError(f'Missing Supply for grade {g}')
        if g not in cost:
            raise ValueError(f'Missing Cost for grade {g}')
    for b in brands:
        if b not in price:
            raise ValueError(f'Missing Price for brand {b}')
    x = m.addVars([(b, g) for b in brands for g in grades], lb=0, vtype=GRB.CONTINUOUS, name='')
    T = m.addVars([b for b in brands], lb=0, vtype=GRB.CONTINUOUS, name='')
    revenue_expr = gurobipy.quicksum((price[b] * T[b] for b in brands))
    cost_expr = gurobipy.quicksum((cost[g] * x[b, g] for b in brands for g in grades))
    m.setObjective(revenue_expr - cost_expr, GRB.MAXIMIZE)
    for (b, g), val in lower_bounds.items():
        m.addConstr(x[b, g] >= val * T[b], name=f'lb_{b}_{g}')
    for (b, g), val in upper_bounds.items():
        m.addConstr(x[b, g] <= val * T[b], name=f'ub_{b}_{g}')
    for g in grades:
        m.addConstr(gurobipy.quicksum((x[b, g] for b in brands)) <= supply[g], name=f'supply_{g}')
    for b in brands:
        m.addConstr(T[b] == gurobipy.quicksum((x[b, g] for g in grades)), name=f't_def_{b}')
    if 'Red' in brands:
        m.addConstr(T['Red'] >= 2000, name='min_red')
    else:
        raise ValueError("Brand 'Red' not found in data")
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()