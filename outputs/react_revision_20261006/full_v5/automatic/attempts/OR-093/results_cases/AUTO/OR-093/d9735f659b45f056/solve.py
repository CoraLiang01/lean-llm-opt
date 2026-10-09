CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A factory has 12 machining machines and 12 tasks. Assigning machine i to task j incurs a machining cost '
          'c_ij, as specified in cost_12x12.csv. The objective is to determine a minimum-cost one-to-one assignment of '
          'machines to tasks, such that each machine is assigned to exactly one task and each task is assigned to '
          'exactly one machine.',
 'relationships': [],
 'route': 'AP',
 'tables': [{'columns': ['Machine', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L'],
             'file_index': 0,
             'file_name': 'cost_12x12.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 12,
             'records': [{'source_row': 0,
                          'values': {'A': '167.4',
                                     'B': '98.6',
                                     'C': '189.4',
                                     'D': '119.6',
                                     'E': '182.0',
                                     'F': '145.1',
                                     'G': '185.4',
                                     'H': '94.8',
                                     'I': '122.3',
                                     'J': '123.3',
                                     'K': '96.1',
                                     'L': '90.3',
                                     'Machine': 'M1'}},
                         {'source_row': 1,
                          'values': {'A': '156.2',
                                     'B': '88.7',
                                     'C': '187.3',
                                     'D': '124.7',
                                     'E': '173.2',
                                     'F': '144.3',
                                     'G': '179.0',
                                     'H': '91.5',
                                     'I': '115.1',
                                     'J': '119.5',
                                     'K': '100.1',
                                     'L': '88.6',
                                     'Machine': 'M2'}},
                         {'source_row': 2,
                          'values': {'A': '184.3',
                                     'B': '121.0',
                                     'C': '216.6',
                                     'D': '140.0',
                                     'E': '196.2',
                                     'F': '168.8',
                                     'G': '205.6',
                                     'H': '114.2',
                                     'I': '133.3',
                                     'J': '144.5',
                                     'K': '116.0',
                                     'L': '107.7',
                                     'Machine': 'M3'}},
                         {'source_row': 3,
                          'values': {'A': '157.9',
                                     'B': '92.9',
                                     'C': '185.1',
                                     'D': '120.3',
                                     'E': '175.1',
                                     'F': '146.2',
                                     'G': '180.8',
                                     'H': '86.3',
                                     'I': '111.6',
                                     'J': '115.9',
                                     'K': '98.1',
                                     'L': '91.1',
                                     'Machine': 'M4'}},
                         {'source_row': 4,
                          'values': {'A': '175.6',
                                     'B': '103.6',
                                     'C': '204.5',
                                     'D': '130.0',
                                     'E': '192.8',
                                     'F': '157.5',
                                     'G': '194.2',
                                     'H': '106.9',
                                     'I': '129.9',
                                     'J': '134.9',
                                     'K': '105.8',
                                     'L': '98.6',
                                     'Machine': 'M5'}},
                         {'source_row': 5,
                          'values': {'A': '166.8',
                                     'B': '107.0',
                                     'C': '199.2',
                                     'D': '130.4',
                                     'E': '183.6',
                                     'F': '159.5',
                                     'G': '187.0',
                                     'H': '98.2',
                                     'I': '121.3',
                                     'J': '126.2',
                                     'K': '105.9',
                                     'L': '101.8',
                                     'Machine': 'M6'}},
                         {'source_row': 6,
                          'values': {'A': '159.7',
                                     'B': '93.2',
                                     'C': '183.8',
                                     'D': '113.0',
                                     'E': '171.9',
                                     'F': '139.1',
                                     'G': '169.6',
                                     'H': '85.1',
                                     'I': '110.0',
                                     'J': '116.7',
                                     'K': '90.6',
                                     'L': '85.2',
                                     'Machine': 'M7'}},
                         {'source_row': 7,
                          'values': {'A': '184.8',
                                     'B': '115.9',
                                     'C': '205.1',
                                     'D': '138.6',
                                     'E': '195.4',
                                     'F': '160.1',
                                     'G': '200.2',
                                     'H': '108.5',
                                     'I': '136.9',
                                     'J': '140.0',
                                     'K': '114.6',
                                     'L': '103.9',
                                     'Machine': 'M8'}},
                         {'source_row': 8,
                          'values': {'A': '157.3',
                                     'B': '86.2',
                                     'C': '186.0',
                                     'D': '113.9',
                                     'E': '166.2',
                                     'F': '136.8',
                                     'G': '167.5',
                                     'H': '78.8',
                                     'I': '107.4',
                                     'J': '114.5',
                                     'K': '87.2',
                                     'L': '78.6',
                                     'Machine': 'M9'}},
                         {'source_row': 9,
                          'values': {'A': '164.8',
                                     'B': '97.8',
                                     'C': '200.9',
                                     'D': '125.8',
                                     'E': '188.9',
                                     'F': '151.2',
                                     'G': '187.7',
                                     'H': '99.5',
                                     'I': '119.5',
                                     'J': '132.1',
                                     'K': '101.1',
                                     'L': '98.4',
                                     'Machine': 'M10'}},
                         {'source_row': 10,
                          'values': {'A': '164.0',
                                     'B': '92.2',
                                     'C': '186.2',
                                     'D': '115.7',
                                     'E': '174.5',
                                     'F': '143.0',
                                     'G': '175.9',
                                     'H': '92.3',
                                     'I': '114.0',
                                     'J': '121.2',
                                     'K': '93.7',
                                     'L': '91.2',
                                     'Machine': 'M11'}},
                         {'source_row': 11,
                          'values': {'A': '151.7',
                                     'B': '76.7',
                                     'C': '179.5',
                                     'D': '109.5',
                                     'E': '160.6',
                                     'F': '128.4',
                                     'G': '170.2',
                                     'H': '74.4',
                                     'I': '103.7',
                                     'J': '110.4',
                                     'K': '83.7',
                                     'L': '75.2',
                                     'Machine': 'M12'}}],
             'returned_rows': 12,
             'role': 'file_0',
             'table_id': 'file_0_view_0'}],
 'validation': {'fallback_reason': "Matrix validation failed: {'matrix_table_id': 'file_0_view_0', 'shape': [12, 12], "
                                   "'expected_shape': [12, 12], 'row_ids_aligned': True, 'column_ids_aligned': False, "
                                   "'row_mapping_basis': 'exact', 'column_mapping_basis': 'unresolved'}",
                'planner_errors': ["Matrix validation failed: {'matrix_table_id': 'file_0_view_0', 'shape': [12, 12], "
                                   "'expected_shape': [12, 12], 'row_ids_aligned': True, 'column_ids_aligned': False, "
                                   "'row_mapping_basis': 'exact', 'column_mapping_basis': 'unresolved'}"],
                'status': 'FALLBACK_FULL_DATA'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    table = None
    for t in CSVQA_DATA['tables']:
        if t['table_id'] == 'file_0_view_0':
            table = t
            break
    if table is None:
        raise ValueError('Cost table not found in CSVQA_DATA.')
    records = table['records']
    row_id_column = 'Machine'
    col_names = [c for c in table['columns'] if c != row_id_column]
    machines = []
    tasks = col_names
    cost = {}
    for rec in records:
        row_id = rec['values'][row_id_column]
        machines.append(row_id)
        cost[row_id] = {}
        for col in col_names:
            val = rec['values'][col]
            try:
                cost[row_id][col] = float(val)
            except Exception:
                raise ValueError(f'Invalid cost value for {row_id}, {col}: {val}')
    if set(cost.keys()) != set(machines):
        raise ValueError('Mismatch in machine identifiers.')
    for m in machines:
        if set(cost[m].keys()) != set(tasks):
            raise ValueError(f'Mismatch in task identifiers for machine {m}.')
    m = gp.Model('FactoryAssignment')
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