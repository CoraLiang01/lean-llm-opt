CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A small courier company operates a single delivery van that must depart from the depot, visit three '
          'customer locations — A, B, and C — exactly once each in any order, and then return to the depot on the same '
          'day. The pairwise road distances (in kilometres) between the depot and every location are provided in '
          'DistanceMatrix.csv. There are no service-time or time-window constraints. Formulate this problem and '
          'determine the sequence of visits that minimises the total travel distance.',
 'relationships': [],
 'route': 'TP',
 'tables': [{'columns': ['Unnamed: 0', 'Depot', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
             'file_index': 0,
             'file_name': 'DistanceMatrix.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 11,
             'records': [{'source_row': 0,
                          'values': {'A': '28',
                                     'B': '41',
                                     'C': '63',
                                     'D': '39',
                                     'Depot': '0',
                                     'E': '38',
                                     'F': '45',
                                     'G': '35',
                                     'H': '28',
                                     'I': '44',
                                     'J': '35',
                                     'Unnamed: 0': 'Depot'}},
                         {'source_row': 1,
                          'values': {'A': '0',
                                     'B': '27',
                                     'C': '87',
                                     'D': '35',
                                     'Depot': '28',
                                     'E': '65',
                                     'F': '63',
                                     'G': '41',
                                     'H': '39',
                                     'I': '43',
                                     'J': '20',
                                     'Unnamed: 0': 'A'}},
                         {'source_row': 2,
                          'values': {'A': '27',
                                     'B': '0',
                                     'C': '81',
                                     'D': '13',
                                     'Depot': '41',
                                     'E': '77',
                                     'F': '54',
                                     'G': '25',
                                     'H': '63',
                                     'I': '70',
                                     'J': '7',
                                     'Unnamed: 0': 'B'}},
                         {'source_row': 3,
                          'values': {'A': '87',
                                     'B': '81',
                                     'C': '0',
                                     'D': '69',
                                     'Depot': '63',
                                     'E': '53',
                                     'F': '28',
                                     'G': '57',
                                     'H': '83',
                                     'I': '102',
                                     'J': '81',
                                     'Unnamed: 0': 'C'}},
                         {'source_row': 4,
                          'values': {'A': '35',
                                     'B': '13',
                                     'C': '69',
                                     'D': '0',
                                     'Depot': '39',
                                     'E': '72',
                                     'F': '41',
                                     'G': '12',
                                     'H': '64',
                                     'I': '75',
                                     'J': '17',
                                     'Unnamed: 0': 'D'}},
                         {'source_row': 5,
                          'values': {'A': '65',
                                     'B': '77',
                                     'C': '53',
                                     'D': '72',
                                     'Depot': '38',
                                     'E': '0',
                                     'F': '53',
                                     'G': '64',
                                     'H': '39',
                                     'I': '58',
                                     'J': '72',
                                     'Unnamed: 0': 'E'}},
                         {'source_row': 6,
                          'values': {'A': '63',
                                     'B': '54',
                                     'C': '28',
                                     'D': '41',
                                     'Depot': '45',
                                     'E': '53',
                                     'F': '0',
                                     'G': '29',
                                     'H': '70',
                                     'I': '88',
                                     'J': '54',
                                     'Unnamed: 0': 'F'}},
                         {'source_row': 7,
                          'values': {'A': '41',
                                     'B': '25',
                                     'C': '57',
                                     'D': '12',
                                     'Depot': '35',
                                     'E': '64',
                                     'F': '29',
                                     'G': '0',
                                     'H': '63',
                                     'I': '76',
                                     'J': '27',
                                     'Unnamed: 0': 'G'}},
                         {'source_row': 8,
                          'values': {'A': '39',
                                     'B': '63',
                                     'C': '83',
                                     'D': '64',
                                     'Depot': '28',
                                     'E': '39',
                                     'F': '70',
                                     'G': '63',
                                     'H': '0',
                                     'I': '20',
                                     'J': '56',
                                     'Unnamed: 0': 'H'}},
                         {'source_row': 9,
                          'values': {'A': '43',
                                     'B': '70',
                                     'C': '102',
                                     'D': '75',
                                     'Depot': '44',
                                     'E': '58',
                                     'F': '88',
                                     'G': '76',
                                     'H': '20',
                                     'I': '0',
                                     'J': '63',
                                     'Unnamed: 0': 'I'}},
                         {'source_row': 10,
                          'values': {'A': '20',
                                     'B': '7',
                                     'C': '81',
                                     'D': '17',
                                     'Depot': '35',
                                     'E': '72',
                                     'F': '54',
                                     'G': '27',
                                     'H': '56',
                                     'I': '63',
                                     'J': '0',
                                     'Unnamed: 0': 'J'}}],
             'returned_rows': 11,
             'role': 'file_0',
             'table_id': 'file_0_view_0'}],
 'validation': {'fallback_reason': "Matrix validation failed: {'matrix_table_id': 'file_0_view_0', 'shape': [4, 4], "
                                   "'expected_shape': [4, 4], 'row_ids_aligned': True, 'column_ids_aligned': False, "
                                   "'row_mapping_basis': 'exact', 'column_mapping_basis': 'unresolved'}",
                'planner_errors': ["Matrix validation failed: {'matrix_table_id': 'file_0_view_0', 'shape': [4, 4], "
                                   "'expected_shape': [4, 4], 'row_ids_aligned': True, 'column_ids_aligned': False, "
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
        raise RuntimeError('Distance matrix table not found.')
    nodes = ['Depot', 'A', 'B', 'C']
    d = {}
    row_map = {rec['values']['Unnamed: 0']: rec for rec in table['records']}
    for i in nodes:
        if i not in row_map:
            raise RuntimeError(f'Missing row for node {i} in distance matrix.')
        rec = row_map[i]
        for j in nodes:
            val = rec['values'][j]
            try:
                d[i, j] = float(val)
            except Exception:
                raise RuntimeError(f'Invalid distance from {i} to {j}: {val}')
    m = gp.Model('TSP_4node')
    x_keys = [(i, j) for i in nodes for j in nodes]
    x = m.addVars(x_keys, vtype=GRB.BINARY, name='')
    u_nodes = ['A', 'B', 'C']
    u = m.addVars(u_nodes, lb=1, ub=3, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((d[i, j] * x[i, j] for i in nodes for j in nodes if i != j)), GRB.MINIMIZE)
    for i in nodes:
        m.addConstr(gp.quicksum((x[i, j] for j in nodes if j != i)) == 1, name=f'depart_{i}')
    for j in nodes:
        m.addConstr(gp.quicksum((x[i, j] for i in nodes if i != j)) == 1, name=f'arrive_{j}')
    for i in u_nodes:
        for j in u_nodes:
            if i != j:
                m.addConstr(u[i] - u[j] + 3 * x[i, j] <= 2, name=f'mtz_{i}_{j}')
    for i in nodes:
        m.addConstr(x[i, i] == 0, name=f'noloop_{i}')
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