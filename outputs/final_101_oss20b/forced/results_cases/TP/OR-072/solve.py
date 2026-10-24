CSVQA_DATA = {'query': 'For a certain daytime and nighttime service bus route, the number of drivers and crew members required during each time period is given in 42.csv:\n\n    Assuming that the drivers and crew members start working at the beginning of each time period and work continuously for 4 hours, how many drivers and crew members should be assigned to this bus route at least? Write down the linear programming model for this problem.', 'relationships': [], 'tables': [{'columns': ['Shift', 'Time', 'Number Required'], 'file_name': '42.csv', 'records': [{'data_row_number': 0, 'values': {'Number Required': '20', 'Shift': '1', 'Time': '0:00-1:00'}}, {'data_row_number': 1, 'values': {'Number Required': '18', 'Shift': '2', 'Time': '1:00-2:00'}}, {'data_row_number': 2, 'values': {'Number Required': '15', 'Shift': '3', 'Time': '2:00-3:00'}}, {'data_row_number': 3, 'values': {'Number Required': '15', 'Shift': '4', 'Time': '3:00-4:00'}}, {'data_row_number': 4, 'values': {'Number Required': '20', 'Shift': '5', 'Time': '4:00-5:00'}}, {'data_row_number': 5, 'values': {'Number Required': '30', 'Shift': '6', 'Time': '5:00-6:00'}}, {'data_row_number': 6, 'values': {'Number Required': '60', 'Shift': '7', 'Time': '6:00-7:00'}}, {'data_row_number': 7, 'values': {'Number Required': '70', 'Shift': '8', 'Time': '7:00-8:00'}}, {'data_row_number': 8, 'values': {'Number Required': '50', 'Shift': '9', 'Time': '8:00-9:00'}}, {'data_row_number': 9, 'values': {'Number Required': '55', 'Shift': '10', 'Time': '9:00-10:00'}}, {'data_row_number': 10, 'values': {'Number Required': '65', 'Shift': '11', 'Time': '10:00-11:00'}}, {'data_row_number': 11, 'values': {'Number Required': '75', 'Shift': '12', 'Time': '11:00-12:00'}}, {'data_row_number': 12, 'values': {'Number Required': '80', 'Shift': '13', 'Time': '12:00-13:00'}}, {'data_row_number': 13, 'values': {'Number Required': '70', 'Shift': '14', 'Time': '13:00-14:00'}}, {'data_row_number': 14, 'values': {'Number Required': '60', 'Shift': '15', 'Time': '14:00-15:00'}}, {'data_row_number': 15, 'values': {'Number Required': '55', 'Shift': '16', 'Time': '15:00-16:00'}}, {'data_row_number': 16, 'values': {'Number Required': '60', 'Shift': '17', 'Time': '16:00-17:00'}}, {'data_row_number': 17, 'values': {'Number Required': '75', 'Shift': '18', 'Time': '17:00-18:00'}}, {'data_row_number': 18, 'values': {'Number Required': '85', 'Shift': '19', 'Time': '18:00-19:00'}}, {'data_row_number': 19, 'values': {'Number Required': '70', 'Shift': '20', 'Time': '19:00-20:00'}}, {'data_row_number': 20, 'values': {'Number Required': '50', 'Shift': '21', 'Time': '20:00-21:00'}}, {'data_row_number': 21, 'values': {'Number Required': '40', 'Shift': '22', 'Time': '21:00-22:00'}}, {'data_row_number': 22, 'values': {'Number Required': '35', 'Shift': '23', 'Time': '22:00-23:00'}}, {'data_row_number': 23, 'values': {'Number Required': '25', 'Shift': '24', 'Time': '23:00-0:00'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    data_table = None
    for tbl in tables:
        if tbl.get('table_id') == 'file_0_view_0':
            data_table = tbl
            break
    if data_table is None:
        raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA")
    records = data_table.get('records', [])
    if not records:
        raise ValueError('No records found in the required table')
    I = set()
    r = {}
    for rec in records:
        values = rec.get('values', {})
        shift_str = values.get('Shift')
        req_str = values.get('Number Required')
        if shift_str is None or req_str is None:
            raise ValueError(f'Missing Shift or Number Required in record {rec}')
        try:
            shift = int(shift_str.strip())
        except Exception as e:
            raise ValueError(f"Invalid Shift value '{shift_str}' in record {rec}") from e
        try:
            req = int(req_str.strip())
        except Exception as e:
            raise ValueError(f"Invalid Number Required value '{req_str}' in record {rec}") from e
        if shift in I:
            raise ValueError(f'Duplicate Shift index {shift} in records')
        I.add(shift)
        r[shift] = req
    expected_shifts = set(range(1, 25))
    if I != expected_shifts:
        missing = expected_shifts - I
        raise ValueError(f'Missing shift indices: {sorted(missing)}')
    m = gurobipy.Model('DriverAssignment')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gurobipy.quicksum((x[i] for i in I)), GRB.MINIMIZE)
    for i in I:
        indices = [(i - k - 1) % 24 + 1 for k in range(4)]
        m.addConstr(gurobipy.quicksum((x[idx] for idx in indices)) >= r[i], name='')
    m.optimize()
    return m
m = solve_problem()
