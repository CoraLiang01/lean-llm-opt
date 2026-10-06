CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A factory has 12 machining machines and 12 tasks. Assigning machine i to task j incurs a machining cost '
          'c_ij, as specified in cost_12x12.csv. The objective is to determine a minimum-cost one-to-one assignment of '
          'machines to tasks, such that each machine is assigned to exactly one task and each task is assigned to '
          'exactly one machine.',
 'relationships': [],
 'route': 'AP',
 'tables': [{'columns': ['assignee_id',
                         'assignment_cost_to_project_A',
                         'assignment_cost_to_project_B',
                         'assignment_cost_to_project_C',
                         'assignment_cost_to_project_D',
                         'assignment_cost_to_project_E',
                         'assignment_cost_to_project_F',
                         'assignment_cost_to_project_G',
                         'assignment_cost_to_project_H',
                         'assignment_cost_to_project_I',
                         'assignment_cost_to_project_J',
                         'assignment_cost_to_project_K',
                         'assignment_cost_to_project_L'],
             'file_index': 0,
             'file_name': 'cost_12x12.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 12,
             'records': [{'source_row': 0,
                          'values': {'assignee_id': 'M1',
                                     'assignment_cost_to_project_A': '167.4',
                                     'assignment_cost_to_project_B': '98.6',
                                     'assignment_cost_to_project_C': '189.4',
                                     'assignment_cost_to_project_D': '119.6',
                                     'assignment_cost_to_project_E': '182.0',
                                     'assignment_cost_to_project_F': '145.1',
                                     'assignment_cost_to_project_G': '185.4',
                                     'assignment_cost_to_project_H': '94.8',
                                     'assignment_cost_to_project_I': '122.3',
                                     'assignment_cost_to_project_J': '123.3',
                                     'assignment_cost_to_project_K': '96.1',
                                     'assignment_cost_to_project_L': '90.3'}},
                         {'source_row': 1,
                          'values': {'assignee_id': 'M2',
                                     'assignment_cost_to_project_A': '156.2',
                                     'assignment_cost_to_project_B': '88.7',
                                     'assignment_cost_to_project_C': '187.3',
                                     'assignment_cost_to_project_D': '124.7',
                                     'assignment_cost_to_project_E': '173.2',
                                     'assignment_cost_to_project_F': '144.3',
                                     'assignment_cost_to_project_G': '179.0',
                                     'assignment_cost_to_project_H': '91.5',
                                     'assignment_cost_to_project_I': '115.1',
                                     'assignment_cost_to_project_J': '119.5',
                                     'assignment_cost_to_project_K': '100.1',
                                     'assignment_cost_to_project_L': '88.6'}},
                         {'source_row': 2,
                          'values': {'assignee_id': 'M3',
                                     'assignment_cost_to_project_A': '184.3',
                                     'assignment_cost_to_project_B': '121.0',
                                     'assignment_cost_to_project_C': '216.6',
                                     'assignment_cost_to_project_D': '140.0',
                                     'assignment_cost_to_project_E': '196.2',
                                     'assignment_cost_to_project_F': '168.8',
                                     'assignment_cost_to_project_G': '205.6',
                                     'assignment_cost_to_project_H': '114.2',
                                     'assignment_cost_to_project_I': '133.3',
                                     'assignment_cost_to_project_J': '144.5',
                                     'assignment_cost_to_project_K': '116.0',
                                     'assignment_cost_to_project_L': '107.7'}},
                         {'source_row': 3,
                          'values': {'assignee_id': 'M4',
                                     'assignment_cost_to_project_A': '157.9',
                                     'assignment_cost_to_project_B': '92.9',
                                     'assignment_cost_to_project_C': '185.1',
                                     'assignment_cost_to_project_D': '120.3',
                                     'assignment_cost_to_project_E': '175.1',
                                     'assignment_cost_to_project_F': '146.2',
                                     'assignment_cost_to_project_G': '180.8',
                                     'assignment_cost_to_project_H': '86.3',
                                     'assignment_cost_to_project_I': '111.6',
                                     'assignment_cost_to_project_J': '115.9',
                                     'assignment_cost_to_project_K': '98.1',
                                     'assignment_cost_to_project_L': '91.1'}},
                         {'source_row': 4,
                          'values': {'assignee_id': 'M5',
                                     'assignment_cost_to_project_A': '175.6',
                                     'assignment_cost_to_project_B': '103.6',
                                     'assignment_cost_to_project_C': '204.5',
                                     'assignment_cost_to_project_D': '130.0',
                                     'assignment_cost_to_project_E': '192.8',
                                     'assignment_cost_to_project_F': '157.5',
                                     'assignment_cost_to_project_G': '194.2',
                                     'assignment_cost_to_project_H': '106.9',
                                     'assignment_cost_to_project_I': '129.9',
                                     'assignment_cost_to_project_J': '134.9',
                                     'assignment_cost_to_project_K': '105.8',
                                     'assignment_cost_to_project_L': '98.6'}},
                         {'source_row': 5,
                          'values': {'assignee_id': 'M6',
                                     'assignment_cost_to_project_A': '166.8',
                                     'assignment_cost_to_project_B': '107.0',
                                     'assignment_cost_to_project_C': '199.2',
                                     'assignment_cost_to_project_D': '130.4',
                                     'assignment_cost_to_project_E': '183.6',
                                     'assignment_cost_to_project_F': '159.5',
                                     'assignment_cost_to_project_G': '187.0',
                                     'assignment_cost_to_project_H': '98.2',
                                     'assignment_cost_to_project_I': '121.3',
                                     'assignment_cost_to_project_J': '126.2',
                                     'assignment_cost_to_project_K': '105.9',
                                     'assignment_cost_to_project_L': '101.8'}},
                         {'source_row': 6,
                          'values': {'assignee_id': 'M7',
                                     'assignment_cost_to_project_A': '159.7',
                                     'assignment_cost_to_project_B': '93.2',
                                     'assignment_cost_to_project_C': '183.8',
                                     'assignment_cost_to_project_D': '113.0',
                                     'assignment_cost_to_project_E': '171.9',
                                     'assignment_cost_to_project_F': '139.1',
                                     'assignment_cost_to_project_G': '169.6',
                                     'assignment_cost_to_project_H': '85.1',
                                     'assignment_cost_to_project_I': '110.0',
                                     'assignment_cost_to_project_J': '116.7',
                                     'assignment_cost_to_project_K': '90.6',
                                     'assignment_cost_to_project_L': '85.2'}},
                         {'source_row': 7,
                          'values': {'assignee_id': 'M8',
                                     'assignment_cost_to_project_A': '184.8',
                                     'assignment_cost_to_project_B': '115.9',
                                     'assignment_cost_to_project_C': '205.1',
                                     'assignment_cost_to_project_D': '138.6',
                                     'assignment_cost_to_project_E': '195.4',
                                     'assignment_cost_to_project_F': '160.1',
                                     'assignment_cost_to_project_G': '200.2',
                                     'assignment_cost_to_project_H': '108.5',
                                     'assignment_cost_to_project_I': '136.9',
                                     'assignment_cost_to_project_J': '140.0',
                                     'assignment_cost_to_project_K': '114.6',
                                     'assignment_cost_to_project_L': '103.9'}},
                         {'source_row': 8,
                          'values': {'assignee_id': 'M9',
                                     'assignment_cost_to_project_A': '157.3',
                                     'assignment_cost_to_project_B': '86.2',
                                     'assignment_cost_to_project_C': '186.0',
                                     'assignment_cost_to_project_D': '113.9',
                                     'assignment_cost_to_project_E': '166.2',
                                     'assignment_cost_to_project_F': '136.8',
                                     'assignment_cost_to_project_G': '167.5',
                                     'assignment_cost_to_project_H': '78.8',
                                     'assignment_cost_to_project_I': '107.4',
                                     'assignment_cost_to_project_J': '114.5',
                                     'assignment_cost_to_project_K': '87.2',
                                     'assignment_cost_to_project_L': '78.6'}},
                         {'source_row': 9,
                          'values': {'assignee_id': 'M10',
                                     'assignment_cost_to_project_A': '164.8',
                                     'assignment_cost_to_project_B': '97.8',
                                     'assignment_cost_to_project_C': '200.9',
                                     'assignment_cost_to_project_D': '125.8',
                                     'assignment_cost_to_project_E': '188.9',
                                     'assignment_cost_to_project_F': '151.2',
                                     'assignment_cost_to_project_G': '187.7',
                                     'assignment_cost_to_project_H': '99.5',
                                     'assignment_cost_to_project_I': '119.5',
                                     'assignment_cost_to_project_J': '132.1',
                                     'assignment_cost_to_project_K': '101.1',
                                     'assignment_cost_to_project_L': '98.4'}},
                         {'source_row': 10,
                          'values': {'assignee_id': 'M11',
                                     'assignment_cost_to_project_A': '164.0',
                                     'assignment_cost_to_project_B': '92.2',
                                     'assignment_cost_to_project_C': '186.2',
                                     'assignment_cost_to_project_D': '115.7',
                                     'assignment_cost_to_project_E': '174.5',
                                     'assignment_cost_to_project_F': '143.0',
                                     'assignment_cost_to_project_G': '175.9',
                                     'assignment_cost_to_project_H': '92.3',
                                     'assignment_cost_to_project_I': '114.0',
                                     'assignment_cost_to_project_J': '121.2',
                                     'assignment_cost_to_project_K': '93.7',
                                     'assignment_cost_to_project_L': '91.2'}},
                         {'source_row': 11,
                          'values': {'assignee_id': 'M12',
                                     'assignment_cost_to_project_A': '151.7',
                                     'assignment_cost_to_project_B': '76.7',
                                     'assignment_cost_to_project_C': '179.5',
                                     'assignment_cost_to_project_D': '109.5',
                                     'assignment_cost_to_project_E': '160.6',
                                     'assignment_cost_to_project_F': '128.4',
                                     'assignment_cost_to_project_G': '170.2',
                                     'assignment_cost_to_project_H': '74.4',
                                     'assignment_cost_to_project_I': '103.7',
                                     'assignment_cost_to_project_J': '110.4',
                                     'assignment_cost_to_project_K': '83.7',
                                     'assignment_cost_to_project_L': '75.2'}}],
             'returned_rows': 12,
             'role': 'cost matrix',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    data = CSVQA_DATA
    table = None
    for t in data['tables']:
        if t['table_id'] == 'file_0_view_0':
            table = t
            break
    if table is None:
        raise ValueError('Cost matrix table not found.')
    records = table['records']
    machines = []
    tasks = []
    cost = {}
    all_columns = table['columns']
    row_id_col = 'assignee_id'
    task_cols = [col for col in all_columns if col != row_id_col]
    tasks = task_cols
    for rec in records:
        machine = rec['values'][row_id_col]
        machines.append(machine)
        cost[machine] = {}
        for task in tasks:
            val = rec['values'][task]
            try:
                cost[machine][task] = float(val)
            except Exception:
                raise ValueError(f'Invalid cost value for {machine}, {task}: {val}')
    if set(cost.keys()) != set(machines):
        raise ValueError('Mismatch in machine identifiers.')
    for m in machines:
        if set(cost[m].keys()) != set(tasks):
            raise ValueError(f'Missing cost entries for machine {m}.')
    m = gp.Model('AssignmentProblem')
    x = m.addVars(machines, tasks, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in machines for j in tasks)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for j in tasks)) == 1 for i in machines), name='')
    m.addConstrs((gp.quicksum((x[i, j] for i in machines)) == 1 for j in tasks), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()