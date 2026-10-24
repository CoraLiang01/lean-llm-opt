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
             'filters': {'conditions': [], 'logic': 'and'},
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
             'filters': {'conditions': [], 'logic': 'and'},
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
import re
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    table0 = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    table1 = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
    supply = {}
    cost = {}
    for rec in table0['records']:
        vals = rec['values']
        grade = vals['Grade'].strip()
        supply[grade] = float(vals['Daily Supply (kg)'])
        cost[grade] = float(vals['Cost (CNY/kg)'])
    price = {}
    lower = {}
    upper = {}
    for rec in table1['records']:
        vals = rec['values']
        brand = vals['Brand'].strip()
        price[brand] = float(vals['Selling Price (CNY/kg)'])
        req = vals['Blending Requirements']
        for match in re.findall('([A-Za-z]+)\\s+(less than|more than)\\s+(\\d+)%', req):
            (grade, bound_type, pct) = match
            grade = grade.strip()
            pct_val = float(pct) / 100.0
            key = (brand, grade)
            if bound_type == 'less than':
                upper[key] = pct_val
            else:
                lower[key] = pct_val
    G = set(supply.keys())
    B = set(price.keys())
    for (b, g) in lower:
        if b not in B or g not in G:
            raise ValueError(f'Lower bound defined for unknown pair ({b},{g})')
    for (b, g) in upper:
        if b not in B or g not in G:
            raise ValueError(f'Upper bound defined for unknown pair ({b},{g})')
    for g in G:
        if g not in supply or g not in cost:
            raise ValueError(f'Missing supply or cost for grade {g}')
    for b in B:
        if b not in price:
            raise ValueError(f'Missing price for brand {b}')
    m = gp.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(g, b) for g in G for b in B], lb=0, vtype=GRB.CONTINUOUS, name='')
    T = {b: gp.quicksum((x[g, b] for g in G)) for b in B}
    obj = gp.quicksum(((price[b] - cost[g]) * x[g, b] for g in G for b in B))
    m.setObjective(obj, GRB.MAXIMIZE)
    for g in G:
        m.addConstr(gp.quicksum((x[g, b] for b in B)) <= supply[g], name=f'supply_con_{g}')
    for ((b, g), lb) in lower.items():
        m.addConstr(x[g, b] >= lb * T[b], name=f'lower_con_{b}_{g}')
    for ((b, g), ub) in upper.items():
        m.addConstr(x[g, b] <= ub * T[b], name=f'upper_con_{b}_{g}')
    if 'Red' in B:
        m.addConstr(T['Red'] >= 2000, name='min_red')
    else:
        raise ValueError("Brand 'Red' not found in data")
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(m.ObjVal)
        for var in m.getVars():
            print(var.VarName, var.X)
    else:
        print(m.Status)
    return m
m = solve_problem()