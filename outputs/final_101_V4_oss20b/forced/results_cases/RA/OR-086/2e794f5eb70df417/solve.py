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
 'route': 'RA',
 'tables': [{'aliases': ['file_0_view_0', '30-1.csv'],
             'columns': ['Grade', 'Daily Supply (kg)', 'Cost (CNY/kg)'],
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
            {'aliases': ['file_1_view_0', '30-2.csv'],
             'columns': ['Brand', 'Blending Requirements', 'Selling Price (CNY/kg)'],
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
import gurobipy as gp
from gurobipy import GRB
grades_table = next((t for t in CSVQA_DATA['tables'] if t['table_id'] == 'file_0_view_0'))
brands_table = next((t for t in CSVQA_DATA['tables'] if t['table_id'] == 'file_1_view_0'))
grades = []
supply = {}
cost = {}
for rec in grades_table['records']:
    g = rec['values']['Grade']
    grades.append(g)
    supply[g] = int(rec['values']['Daily Supply (kg)'])
    cost[g] = float(rec['values']['Cost (CNY/kg)'])
brands = []
selling_price = {}
for rec in brands_table['records']:
    b = rec['values']['Brand']
    brands.append(b)
    selling_price[b] = float(rec['values']['Selling Price (CNY/kg)'])
m = gp.Model('wine_blending')
x = {}
for g in grades:
    x[g] = {}
    for b in brands:
        x[g][b] = m.addVar(vtype=GRB.INTEGER)
m.update()
obj = gp.quicksum((selling_price[b] * x[g][b] for g in grades for b in brands)) - gp.quicksum((cost[g] * x[g][b] for g in grades for b in brands))
m.setObjective(obj, GRB.MAXIMIZE)
m.addConstr(9 * x['I']['Red'] <= x['II']['Red'] + x['III']['Red'])
m.addConstr(x['II']['Red'] >= x['I']['Red'] + x['III']['Red'])
m.addConstr(3 * x['III']['Yellow'] <= 7 * x['I']['Yellow'] + 7 * x['II']['Yellow'])
m.addConstr(4 * x['I']['Yellow'] >= x['II']['Yellow'] + x['III']['Yellow'])
m.addConstr(x['I']['Blue'] <= x['II']['Blue'] + x['III']['Blue'])
m.addConstr(9 * x['II']['Blue'] >= x['I']['Blue'] + x['III']['Blue'])
for g in grades:
    m.addConstr(gp.quicksum((x[g][b] for b in brands)) <= supply[g])
m.addConstr(gp.quicksum((x[g]['Red'] for g in grades)) >= 2000)
m.Params.MIPGap = 1e-06
m.optimize()
if m.status == GRB.OPTIMAL:
    print(f'Objective value: {m.ObjVal}')
    for g in grades:
        for b in brands:
            var = x[g][b]
            print(f'{var.VarName} = {var.X}')
else:
    print(f'Solver status: {m.status}')