CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'For a certain daytime and nighttime service bus route, the number of drivers and crew members required '
          'during each time period is given in 42.csv:\n'
          '\n'
          '    Assuming that the drivers and crew members start working at the beginning of each time period and work '
          'continuously for 4 hours, how many drivers and crew members should be assigned to this bus route at least? '
          'Write down the linear programming model for this problem.',
 'relationships': [],
 'route': 'FLP',
 'tables': [{'columns': ['Shift', 'Time', 'Number Required'],
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
import gurobipy
from gurobipy import GRB

def solve_problem():
    table = None
    for tbl in CSVQA_DATA['tables']:
        if tbl.get('table_id') == 'file_0_view_0':
            table = tbl
            break
    if table is None:
        raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA")
    shifts = set()
    r_s = {}
    for record in table['records']:
        values = record['values']
        shift_str = values.get('Shift')
        req_str = values.get('Number Required')
        if shift_str is None or req_str is None:
            raise ValueError(f'Missing Shift or Number Required in record {record}')
        try:
            shift = int(shift_str)
            req = int(req_str)
        except ValueError:
            raise ValueError(f'Non-integer Shift or Number Required in record {record}')
        if shift in shifts:
            raise ValueError(f'Duplicate shift value {shift} in data')
        shifts.add(shift)
        r_s[shift] = req
    expected_shifts = set(range(1, 25))
    if shifts != expected_shifts:
        missing = expected_shifts - shifts
        extra = shifts - expected_shifts
        msg = ''
        if missing:
            msg += f'Missing shifts: {sorted(missing)}. '
        if extra:
            msg += f'Unexpected shifts: {sorted(extra)}.'
        raise ValueError(msg)
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(shifts, lb=0, vtype=GRB.INTEGER, name='')
    model.setObjective(gurobipy.quicksum((x[s] for s in shifts)), GRB.MINIMIZE)
    for s in shifts:
        expr = gurobipy.quicksum((x[(s - k - 1) % 24 + 1] for k in range(4)))
        model.addConstr(expr >= r_s[s], name='')
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(f'ObjVal: {model.ObjVal}')
        for var in model.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {model.Status}')
    return model
m = solve_problem()