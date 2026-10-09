CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A fabrication shop must assign custom jobs to workstations. Workstation capacities are listed in '
          'workstation_capacity.csv, assignment costs are listed in assignment_costs.csv, and the capacity consumed by '
          'each workstation-job assignment is listed in assignment_resources.csv. Each job must be assigned to exactly '
          'one workstation.\n'
          '\n'
          'Formulate a minimum-cost generalized assignment model. For each workstation-job pair i-j, define x_ij as a '
          'binary variable equal to 1 if job j is assigned to workstation i. The objective is to minimize total '
          'assignment cost. The model should include exactly-one assignment constraints for every job, capacity '
          'constraints for every workstation using the listed resource consumption coefficients, and binary '
          'restrictions for all assignment variables.',
 'relationships': [],
 'route': 'AP',
 'tables': [{'columns': ['Workstation', 'Capacity'],
             'file_index': 0,
             'file_name': 'workstation_capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 4,
             'records': [{'source_row': 0, 'values': {'Capacity': '15', 'Workstation': 'W1'}},
                         {'source_row': 1, 'values': {'Capacity': '14', 'Workstation': 'W2'}},
                         {'source_row': 2, 'values': {'Capacity': '16', 'Workstation': 'W3'}},
                         {'source_row': 3, 'values': {'Capacity': '13', 'Workstation': 'W4'}}],
             'returned_rows': 4,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['Workstation', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9'],
             'file_index': 1,
             'file_name': 'assignment_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 4,
             'records': [{'source_row': 0,
                          'values': {'J1': '6',
                                     'J2': '8',
                                     'J3': '18',
                                     'J4': '20',
                                     'J5': '21',
                                     'J6': '19',
                                     'J7': '23',
                                     'J8': '22',
                                     'J9': '24',
                                     'Workstation': 'W1'}},
                         {'source_row': 1,
                          'values': {'J1': '19',
                                     'J2': '18',
                                     'J3': '7',
                                     'J4': '6',
                                     'J5': '20',
                                     'J6': '22',
                                     'J7': '21',
                                     'J8': '23',
                                     'J9': '25',
                                     'Workstation': 'W2'}},
                         {'source_row': 2,
                          'values': {'J1': '22',
                                     'J2': '21',
                                     'J3': '20',
                                     'J4': '19',
                                     'J5': '5',
                                     'J6': '7',
                                     'J7': '18',
                                     'J8': '20',
                                     'J9': '21',
                                     'Workstation': 'W3'}},
                         {'source_row': 3,
                          'values': {'J1': '21',
                                     'J2': '22',
                                     'J3': '23',
                                     'J4': '20',
                                     'J5': '19',
                                     'J6': '18',
                                     'J7': '6',
                                     'J8': '8',
                                     'J9': '7',
                                     'Workstation': 'W4'}}],
             'returned_rows': 4,
             'role': 'file_1',
             'table_id': 'file_1_view_0'},
            {'columns': ['Workstation', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9'],
             'file_index': 2,
             'file_name': 'assignment_resources.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 4,
             'records': [{'source_row': 0,
                          'values': {'J1': '4',
                                     'J2': '5',
                                     'J3': '7',
                                     'J4': '8',
                                     'J5': '8',
                                     'J6': '7',
                                     'J7': '8',
                                     'J8': '9',
                                     'J9': '8',
                                     'Workstation': 'W1'}},
                         {'source_row': 1,
                          'values': {'J1': '7',
                                     'J2': '8',
                                     'J3': '4',
                                     'J4': '5',
                                     'J5': '8',
                                     'J6': '8',
                                     'J7': '7',
                                     'J8': '8',
                                     'J9': '9',
                                     'Workstation': 'W2'}},
                         {'source_row': 2,
                          'values': {'J1': '8',
                                     'J2': '7',
                                     'J3': '8',
                                     'J4': '7',
                                     'J5': '5',
                                     'J6': '4',
                                     'J7': '7',
                                     'J8': '8',
                                     'J9': '7',
                                     'Workstation': 'W3'}},
                         {'source_row': 3,
                          'values': {'J1': '8',
                                     'J2': '8',
                                     'J3': '9',
                                     'J4': '8',
                                     'J5': '7',
                                     'J6': '7',
                                     'J7': '4',
                                     'J8': '5',
                                     'J9': '4',
                                     'Workstation': 'W4'}}],
             'returned_rows': 4,
             'role': 'file_2',
             'table_id': 'file_2_view_0'}],
 'validation': {'fallback_reason': "Matrix validation failed: {'matrix_table_id': 'file_1_view_0', 'shape': [4, 9], "
                                   "'expected_shape': [4, 4], 'row_ids_aligned': True, 'column_ids_aligned': False, "
                                   "'row_mapping_basis': 'exact', 'column_mapping_basis': 'unresolved'}",
                'planner_errors': ["Matrix validation failed: {'matrix_table_id': 'file_1_view_0', 'shape': [4, 9], "
                                   "'expected_shape': [4, 4], 'row_ids_aligned': True, 'column_ids_aligned': False, "
                                   "'row_mapping_basis': 'exact', 'column_mapping_basis': 'unresolved'}"],
                'status': 'FALLBACK_FULL_DATA'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem(CSVQA_FRAMES):
    cap_frame = CSVQA_FRAMES['file_0_view_0']
    workstations = []
    b = {}
    for (_, row) in cap_frame.iterrows():
        ws = row['Workstation']
        workstations.append(ws)
        try:
            b[ws] = float(row['Capacity'])
        except Exception:
            raise ValueError(f"Invalid capacity for workstation {ws}: {row['Capacity']}")
    cost_frame = CSVQA_FRAMES['file_1_view_0']
    jobs = [col for col in cost_frame.columns if col != 'Workstation']
    cost_ws = list(cost_frame['Workstation'])
    if set(workstations) != set(cost_ws):
        raise ValueError('Mismatch in workstation sets between capacity and cost data.')
    res_frame = CSVQA_FRAMES['file_2_view_0']
    res_ws = list(res_frame['Workstation'])
    if set(workstations) != set(res_ws):
        raise ValueError('Mismatch in workstation sets between capacity and resource data.')
    c = {}
    a = {}
    for (_, row) in cost_frame.iterrows():
        ws = row['Workstation']
        c[ws] = {}
        for j in jobs:
            try:
                c[ws][j] = float(row[j])
            except Exception:
                raise ValueError(f'Invalid assignment cost for ({ws},{j}): {row[j]}')
    for (_, row) in res_frame.iterrows():
        ws = row['Workstation']
        a[ws] = {}
        for j in jobs:
            try:
                a[ws][j] = float(row[j])
            except Exception:
                raise ValueError(f'Invalid assignment resource for ({ws},{j}): {row[j]}')
    for ws in workstations:
        if ws not in c or ws not in a:
            raise ValueError(f'Missing data for workstation {ws}')
        if set(c[ws].keys()) != set(jobs) or set(a[ws].keys()) != set(jobs):
            raise ValueError(f'Missing job data for workstation {ws}')
    m = gp.Model('GeneralizedAssignment')
    m.setParam('MIPGap', 0.0001)
    x_keys = [(i, j) for i in workstations for j in jobs]
    x_vars = m.addVars(x_keys, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((c[i][j] * x_vars[i, j] for i in workstations for j in jobs)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x_vars[i, j] for i in workstations)) == 1 for j in jobs), name='')
    m.addConstrs((gp.quicksum((a[i][j] * x_vars[i, j] for j in jobs)) <= b[i] for i in workstations), name='')
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')