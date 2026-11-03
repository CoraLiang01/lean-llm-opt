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
 'route': 'Others',
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

def solve_problem():
    tables = CSVQA_DATA['tables']
    table0 = next((t for t in tables if t['file_index'] == 0))
    grades = []
    supply = {}
    cost = {}
    for rec in table0['records']:
        grade = rec['values']['Grade']
        grades.append(grade)
        supply[grade] = float(rec['values']['Daily Supply (kg)'])
        cost[grade] = float(rec['values']['Cost (CNY/kg)'])
    table1 = next((t for t in tables if t['file_index'] == 1))
    brands = []
    price = {}
    lower = {}
    upper = {}
    for rec in table1['records']:
        brand = rec['values']['Brand']
        brands.append(brand)
        price[brand] = float(rec['values']['Selling Price (CNY/kg)'])
        req_str = rec['values']['Blending Requirements']
        pattern = '([A-Za-z]+)\\s+(less than|more than)\\s+([0-9]+(?:\\.[0-9]+)?)%'
        matches = re.findall(pattern, req_str)
        lower[brand] = {}
        upper[brand] = {}
        for g, rel, pct in matches:
            val = float(pct) / 100.0
            if rel == 'more than':
                lower[brand][g] = val
            else:
                upper[brand][g] = val
    for g in grades:
        if g not in supply or g not in cost:
            raise ValueError(f'Missing data for grade {g}')
    for b in brands:
        if b not in price:
            raise ValueError(f'Missing price for brand {b}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(g, b) for g in grades for b in brands], lb=0, vtype=GRB.CONTINUOUS, name='')
    y = m.addVars(brands, lb=0, vtype=GRB.CONTINUOUS, name='')
    revenue = gurobipy.quicksum((price[b] * y[b] for b in brands))
    cost_expr = gurobipy.quicksum((cost[g] * x[g, b] for g in grades for b in brands))
    m.setObjective(revenue - cost_expr, GRB.MAXIMIZE)
    for b in brands:
        m.addConstr(y[b] == gurobipy.quicksum((x[g, b] for g in grades)), name=f'prod_def_{b}')
    for g in grades:
        m.addConstr(gurobipy.quicksum((x[g, b] for b in brands)) <= supply[g], name=f'supply_{g}')
    for b in brands:
        for g in grades:
            if g in lower[b]:
                m.addConstr(x[g, b] >= lower[b][g] * y[b], name=f'lower_{b}_{g}')
            if g in upper[b]:
                m.addConstr(x[g, b] <= upper[b][g] * y[b], name=f'upper_{b}_{g}')
    if 'Red' in brands:
        m.addConstr(y['Red'] >= 2000, name='min_prod_Red')
    else:
        raise ValueError('Brand Red not found')
    m.optimize()
    return m
m = solve_problem()