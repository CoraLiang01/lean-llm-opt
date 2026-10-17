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
 'route': 'AP',
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
    data = CSVQA_DATA
    tables = data['tables']
    table0 = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    table1 = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
    G = []
    cost_g = {}
    supply_g = {}
    for rec in table0['records']:
        grade = rec['values']['Grade'].strip()
        G.append(grade)
        cost_g[grade] = float(rec['values']['Cost (CNY/kg)'])
        supply_g[grade] = float(rec['values']['Daily Supply (kg)'])
    B = []
    price_b = {}
    lower_bounds = {}
    upper_bounds = {}
    for rec in table1['records']:
        brand = rec['values']['Brand'].strip()
        B.append(brand)
        price_b[brand] = float(rec['values']['Selling Price (CNY/kg)'])
        req_str = rec['values']['Blending Requirements']
        pattern = '([A-Z]+)\\s+(less than|more than)\\s+(\\d+)%'
        for m in re.finditer(pattern, req_str):
            grade = m.group(1)
            relation = m.group(2)
            pct = int(m.group(3)) / 100.0
            key = (grade, brand)
            if relation == 'less than':
                upper_bounds[key] = pct
            else:
                lower_bounds[key] = pct
    for g in G:
        if g not in cost_g or g not in supply_g:
            raise ValueError(f'Missing cost or supply for grade {g}')
    for b in B:
        if b not in price_b:
            raise ValueError(f'Missing price for brand {b}')
    for g, b in lower_bounds:
        if g not in G or b not in B:
            raise ValueError(f'Lower bound refers to unknown pair {g},{b}')
    for g, b in upper_bounds:
        if g not in G or b not in B:
            raise ValueError(f'Upper bound refers to unknown pair {g},{b}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars([(g, b) for g in G for b in B], vtype=GRB.CONTINUOUS, lb=0, name='')
    y = model.addVars(B, vtype=GRB.CONTINUOUS, lb=0, name='')
    for (g, b), lb in lower_bounds.items():
        model.addConstr(x[g, b] >= lb * y[b], name=f'lb_{g}_{b}')
    for (g, b), ub in upper_bounds.items():
        model.addConstr(x[g, b] <= ub * y[b], name=f'ub_{g}_{b}')
    for g in G:
        model.addConstr(gurobipy.quicksum((x[g, b] for b in B)) <= supply_g[g], name=f'supply_{g}')
    if 'Red' not in B:
        raise ValueError('Brand Red not found')
    model.addConstr(y['Red'] >= 2000, name='min_red')
    revenue = gurobipy.quicksum((price_b[b] * y[b] for b in B))
    cost = gurobipy.quicksum((cost_g[g] * gurobipy.quicksum((x[g, b] for b in B)) for g in G))
    model.setObjective(revenue - cost, GRB.MAXIMIZE)
    model.optimize()
    return model
m = solve_problem()