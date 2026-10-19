CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'For a certain daytime and nighttime service bus route, the number of drivers and crew members required '
          'during each time period is given in 42.csv:\n'
          '\n'
          '    Assuming that the drivers and crew members start working at the beginning of each time period and work '
          'continuously for 4 hours, how many drivers and crew members should be assigned to this bus route at least? '
          'Write down the linear programming model for this problem.',
 'relationships': [],
 'route': 'TP',
 'tables': [{'aliases': ['file_0_view_0', '42.csv'],
             'columns': ['Shift', 'Time', 'Number Required'],
             'file_index': 0,
             'file_name': '42.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 24,
             'records': [{'source_row': 0, 'values': {'Number Required': '20', 'Shift': '1', 'Time': '0:00-1:00'}},
                         {'source_row': 1, 'values': {'Number Required': '18', 'Shift': '2', 'Time': '1:00-2:00'}},
                         {'source_row': 2, 'values': {'Number Required': '15', 'Shift': '3', 'Time': '2:00-3:00'}},
                         {'source_row': 3, 'values': {'Number Required': '15', 'Shift': '4', 'Time': '3:00-4:00'}},
                         {'source_row': 4, 'values': {'Number Required': '20', 'Shift': '5', 'Time': '4:00-5:00'}},
                         {'source_row': 5, 'values': {'Number Required': '30', 'Shift': '6', 'Time': '5:00-6:00'}},
                         {'source_row': 6, 'values': {'Number Required': '60', 'Shift': '7', 'Time': '6:00-7:00'}},
                         {'source_row': 7, 'values': {'Number Required': '70', 'Shift': '8', 'Time': '7:00-8:00'}},
                         {'source_row': 8, 'values': {'Number Required': '50', 'Shift': '9', 'Time': '8:00-9:00'}},
                         {'source_row': 9, 'values': {'Number Required': '55', 'Shift': '10', 'Time': '9:00-10:00'}},
                         {'source_row': 10, 'values': {'Number Required': '65', 'Shift': '11', 'Time': '10:00-11:00'}},
                         {'source_row': 11, 'values': {'Number Required': '75', 'Shift': '12', 'Time': '11:00-12:00'}},
                         {'source_row': 12, 'values': {'Number Required': '80', 'Shift': '13', 'Time': '12:00-13:00'}},
                         {'source_row': 13, 'values': {'Number Required': '70', 'Shift': '14', 'Time': '13:00-14:00'}},
                         {'source_row': 14, 'values': {'Number Required': '60', 'Shift': '15', 'Time': '14:00-15:00'}},
                         {'source_row': 15, 'values': {'Number Required': '55', 'Shift': '16', 'Time': '15:00-16:00'}},
                         {'source_row': 16, 'values': {'Number Required': '60', 'Shift': '17', 'Time': '16:00-17:00'}},
                         {'source_row': 17, 'values': {'Number Required': '75', 'Shift': '18', 'Time': '17:00-18:00'}},
                         {'source_row': 18, 'values': {'Number Required': '85', 'Shift': '19', 'Time': '18:00-19:00'}},
                         {'source_row': 19, 'values': {'Number Required': '70', 'Shift': '20', 'Time': '19:00-20:00'}},
                         {'source_row': 20, 'values': {'Number Required': '50', 'Shift': '21', 'Time': '20:00-21:00'}},
                         {'source_row': 21, 'values': {'Number Required': '40', 'Shift': '22', 'Time': '21:00-22:00'}},
                         {'source_row': 22, 'values': {'Number Required': '35', 'Shift': '23', 'Time': '22:00-23:00'}},
                         {'source_row': 23, 'values': {'Number Required': '25', 'Shift': '24', 'Time': '23:00-0:00'}}],
             'returned_rows': 24,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB
required = {}
for rec in CSVQA_DATA['tables'][0]['records']:
    shift = int(rec['values']['Shift'])
    req = int(rec['values']['Number Required'])
    required[shift] = req
m = gp.Model()
m.setParam('MIPGap', 1e-06)
x = {}
for t in range(1, 22):
    x[t] = m.addVar(vtype=GRB.INTEGER)
m.update()
m.setObjective(gp.quicksum((x[t] for t in range(1, 22))), GRB.MINIMIZE)
for t in range(1, 22):
    start = max(1, t - 3)
    expr = gp.quicksum((x[i] for i in range(start, t + 1)))
    m.addConstr(expr >= required[t])
m.addConstr(x[19] + x[20] + x[21] >= required[22])
m.addConstr(x[20] + x[21] >= required[23])
m.addConstr(x[21] >= required[24])
m.optimize()
if m.status == GRB.OPTIMAL:
    print('Objective:', m.objVal)
    for t in range(1, 22):
        print(f'x{t} =', x[t].X)
else:
    print('Solver status:', m.status)