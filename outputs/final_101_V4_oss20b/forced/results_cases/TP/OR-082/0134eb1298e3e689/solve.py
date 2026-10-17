CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A small courier company operates a single delivery van that must depart from the depot, visit three '
          'customer locations — A, B, and C — exactly once each in any order, and then return to the depot on the same '
          'day. The pairwise road distances (in kilometres) between the depot and every location are provided in '
          'DistanceMatrix.csv. There are no service-time or time-window constraints. Formulate this problem and '
          'determine the sequence of visits that minimises the total travel distance.',
 'relationships': [{'column_axis': {'id_column': 'Unnamed: 0', 'table_id': 'file_0_view_0'},
                    'matrix_table_id': 'file_0_view_0',
                    'row_axis': {'id_column': 'Unnamed: 0', 'table_id': 'file_0_view_0'},
                    'row_id_column': 'Unnamed: 0',
                    'type': 'matrix'}],
 'route': 'TP',
 'tables': [{'columns': ['Unnamed: 0', 'Depot', 'A', 'B', 'C'],
             'file_index': 0,
             'file_name': 'DistanceMatrix.csv',
             'filters': {'conditions': [{'column': 'Unnamed: 0',
                                         'dtype': 'string',
                                         'evidence': 'Depot, A, B, C',
                                         'operator': 'in',
                                         'value': ['Depot', 'A', 'B', 'C']}],
                         'logic': 'and'},
             'original_rows': 11,
             'records': [{'source_row': 0,
                          'values': {'A': '28', 'B': '41', 'C': '63', 'Depot': '0', 'Unnamed: 0': 'Depot'}},
                         {'source_row': 1,
                          'values': {'A': '0', 'B': '27', 'C': '87', 'Depot': '28', 'Unnamed: 0': 'A'}},
                         {'source_row': 2,
                          'values': {'A': '27', 'B': '0', 'C': '81', 'Depot': '41', 'Unnamed: 0': 'B'}},
                         {'source_row': 3,
                          'values': {'A': '87', 'B': '81', 'C': '0', 'Depot': '63', 'Unnamed: 0': 'C'}}],
             'returned_rows': 4,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [{'column_ids_aligned': True,
                                   'column_mapping_basis': 'exact',
                                   'expected_shape': [4, 4],
                                   'matrix_table_id': 'file_0_view_0',
                                   'row_ids_aligned': True,
                                   'row_mapping_basis': 'exact',
                                   'shape': [4, 4]}],
                'status': 'OK'}}
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
    records = table['records']
    columns = table['columns']
    N = {rec['values']['Unnamed: 0'] for rec in records}
    V = N - {'Depot'}
    d = {}
    for rec in records:
        i = rec['values']['Unnamed: 0']
        for col in columns:
            if col == 'Unnamed: 0':
                continue
            j = col
            if i == j:
                continue
            val_str = rec['values'][col]
            try:
                val = float(val_str)
            except ValueError:
                raise ValueError(f'Non‑numeric distance value for pair ({i},{j}): {val_str}')
            d[i, j] = val
    expected_pairs = [(i, j) for i in N for j in N if i != j]
    missing = [pair for pair in expected_pairs if pair not in d]
    if missing:
        raise ValueError(f'Missing distance entries for pairs: {missing}')
    m = gurobipy.Model('TSP')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(i, j) for i in N for j in N if i != j], vtype=GRB.BINARY, name='')
    u = m.addVars([i for i in V], vtype=GRB.CONTINUOUS, lb=1, ub=len(V), name='')
    m.setObjective(gurobipy.quicksum((d[i, j] * x[i, j] for i in N for j in N if i != j)), GRB.MINIMIZE)
    for i in N:
        m.addConstr(gurobipy.quicksum((x[i, j] for j in N if j != i)) == 1, name=f'out_{i}')
    for j in N:
        m.addConstr(gurobipy.quicksum((x[i, j] for i in N if i != j)) == 1, name=f'in_{j}')
    for i in V:
        for j in V:
            if i == j:
                continue
            m.addConstr(u[i] - u[j] + len(V) * x[i, j] <= len(V) - 1, name=f'mtz_{i}_{j}')
    m.optimize()
    return m
m = solve_problem()