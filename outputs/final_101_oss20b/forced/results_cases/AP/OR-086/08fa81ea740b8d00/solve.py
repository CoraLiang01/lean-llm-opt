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
grades = ['I', 'II', 'III']
brands = ['Red', 'Yellow', 'Blue']
supply = {'I': 1500, 'II': 2000, 'III': 1000}
cost = {'I': 6, 'II': 4.5, 'III': 3}
price = {'Red': 5.5, 'Yellow': 5, 'Blue': 4.8}
m = gp.Model('Wine_Production')
x = m.addVars(grades, brands, vtype=GRB.CONTINUOUS)
obj = gp.quicksum((price[b] * x[g, b] - cost[g] * x[g, b] for g in grades for b in brands))
m.setObjective(obj, GRB.MAXIMIZE)
for g in grades:
    m.addConstr(gp.quicksum((x[g, b] for b in brands)) <= supply[g])
m.addConstr(gp.quicksum((x[g, 'Red'] for g in grades)) >= 2000)
total_red = gp.quicksum((x[g, 'Red'] for g in grades))
m.addConstr(x['I', 'Red'] <= 0.1 * total_red)
m.addConstr(x['II', 'Red'] >= 0.5 * total_red)
total_yellow = gp.quicksum((x[g, 'Yellow'] for g in grades))
m.addConstr(x['III', 'Yellow'] <= 0.7 * total_yellow)
m.addConstr(x['I', 'Yellow'] >= 0.2 * total_yellow)
total_blue = gp.quicksum((x[g, 'Blue'] for g in grades))
m.addConstr(x['I', 'Blue'] <= 0.5 * total_blue)
m.addConstr(x['II', 'Blue'] >= 0.1 * total_blue)
m.Params.MIPGap = 1e-06
m.optimize()
if m.status == GRB.OPTIMAL:
    print('Optimal objective:', m.objVal)
    for g in grades:
        for b in brands:
            val = x[g, b].X
            if val > 1e-06:
                print(f'x[{g},{b}] = {val}')
else:
    print('Solver status:', m.status)