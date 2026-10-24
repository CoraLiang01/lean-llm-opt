CSVQA_DATA = {'query': 'A fabrication shop must assign custom jobs to workstations. Workstation capacities are listed in workstation_capacity.csv, assignment costs are listed in assignment_costs.csv, and the capacity consumed by each workstation-job assignment is listed in assignment_resources.csv. Each job must be assigned to exactly one workstation.\n\nFormulate a minimum-cost generalized assignment model. For each workstation-job pair i-j, define x_ij as a binary variable equal to 1 if job j is assigned to workstation i. The objective is to minimize total assignment cost. The model should include exactly-one assignment constraints for every job, capacity constraints for every workstation using the listed resource consumption coefficients, and binary restrictions for all assignment variables.', 'relationships': [], 'tables': [{'columns': ['Workstation', 'Capacity'], 'file_name': 'workstation_capacity.csv', 'records': [{'data_row_number': 0, 'values': {'Capacity': '15', 'Workstation': 'W1'}}, {'data_row_number': 1, 'values': {'Capacity': '14', 'Workstation': 'W2'}}, {'data_row_number': 2, 'values': {'Capacity': '16', 'Workstation': 'W3'}}, {'data_row_number': 3, 'values': {'Capacity': '13', 'Workstation': 'W4'}}], 'table_id': 'file_0_view_0'}, {'columns': ['Workstation', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9'], 'file_name': 'assignment_costs.csv', 'records': [{'data_row_number': 0, 'values': {'J1': '6', 'J2': '8', 'J3': '18', 'J4': '20', 'J5': '21', 'J6': '19', 'J7': '23', 'J8': '22', 'J9': '24', 'Workstation': 'W1'}}, {'data_row_number': 1, 'values': {'J1': '19', 'J2': '18', 'J3': '7', 'J4': '6', 'J5': '20', 'J6': '22', 'J7': '21', 'J8': '23', 'J9': '25', 'Workstation': 'W2'}}, {'data_row_number': 2, 'values': {'J1': '22', 'J2': '21', 'J3': '20', 'J4': '19', 'J5': '5', 'J6': '7', 'J7': '18', 'J8': '20', 'J9': '21', 'Workstation': 'W3'}}, {'data_row_number': 3, 'values': {'J1': '21', 'J2': '22', 'J3': '23', 'J4': '20', 'J5': '19', 'J6': '18', 'J7': '6', 'J8': '8', 'J9': '7', 'Workstation': 'W4'}}], 'table_id': 'file_1_view_0'}, {'columns': ['Workstation', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9'], 'file_name': 'assignment_resources.csv', 'records': [{'data_row_number': 0, 'values': {'J1': '4', 'J2': '5', 'J3': '7', 'J4': '8', 'J5': '8', 'J6': '7', 'J7': '8', 'J8': '9', 'J9': '8', 'Workstation': 'W1'}}, {'data_row_number': 1, 'values': {'J1': '7', 'J2': '8', 'J3': '4', 'J4': '5', 'J5': '8', 'J6': '8', 'J7': '7', 'J8': '8', 'J9': '9', 'Workstation': 'W2'}}, {'data_row_number': 2, 'values': {'J1': '8', 'J2': '7', 'J3': '8', 'J4': '7', 'J5': '5', 'J6': '4', 'J7': '7', 'J8': '8', 'J9': '7', 'Workstation': 'W3'}}, {'data_row_number': 3, 'values': {'J1': '8', 'J2': '8', 'J3': '9', 'J4': '8', 'J5': '7', 'J6': '7', 'J7': '4', 'J8': '5', 'J9': '4', 'Workstation': 'W4'}}], 'table_id': 'file_2_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    table0 = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    table1 = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
    table2 = next((t for t in tables if t['table_id'] == 'file_2_view_0'))
    c_i = {}
    for rec in table0['records']:
        ws = rec['values']['Workstation']
        cap = int(rec['values']['Capacity'])
        c_i[ws] = cap
    jobs = [col for col in table1['columns'] if col != 'Workstation']
    a_ij = {}
    for rec in table1['records']:
        ws = rec['values']['Workstation']
        for job in jobs:
            a_ij[ws, job] = int(rec['values'][job])
    r_ij = {}
    for rec in table2['records']:
        ws = rec['values']['Workstation']
        for job in jobs:
            r_ij[ws, job] = int(rec['values'][job])
    workstations = list(c_i.keys())
    for ws in workstations:
        if ws not in c_i:
            raise ValueError(f'Missing capacity for workstation {ws}')
        for job in jobs:
            if (ws, job) not in a_ij:
                raise ValueError(f'Missing cost for workstation {ws} job {job}')
            if (ws, job) not in r_ij:
                raise ValueError(f'Missing resource for workstation {ws} job {job}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(ws, job) for ws in workstations for job in jobs], vtype=GRB.BINARY, name='')
    m.setObjective(gurobipy.quicksum((a_ij[ws, job] * x[ws, job] for ws in workstations for job in jobs)), GRB.MINIMIZE)
    m.addConstrs((gurobipy.quicksum((x[ws, job] for ws in workstations)) == 1 for job in jobs), name='')
    m.addConstrs((gurobipy.quicksum((r_ij[ws, job] * x[ws, job] for job in jobs)) <= c_i[ws] for ws in workstations), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()
