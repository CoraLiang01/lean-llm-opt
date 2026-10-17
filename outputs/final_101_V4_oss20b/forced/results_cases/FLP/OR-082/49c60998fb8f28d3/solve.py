CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A small courier company operates a single delivery van that must depart from the depot, visit three '
          'customer locations — A, B, and C — exactly once each in any order, and then return to the depot on the same '
          'day. The pairwise road distances (in kilometres) between the depot and every location are provided in '
          'DistanceMatrix.csv. There are no service-time or time-window constraints. Formulate this problem and '
          'determine the sequence of visits that minimises the total travel distance.',
 'relationships': [],
 'route': 'FLP',
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
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table = next((t for t in CSVQA_DATA['tables'] if t['table_id'] == 'file_0_view_0'))
    columns = table['columns']
    records = table['records']
    row_labels = {rec['values']['Unnamed: 0'] for rec in records}
    col_labels = set(columns[1:])
    N = row_labels.union(col_labels)
    d = {}
    for rec in records:
        i = rec['values']['Unnamed: 0']
        for j in columns[1:]:
            if i == j:
                continue
            val_str = rec['values'][j]
            try:
                val = float(val_str)
            except ValueError:
                raise ValueError(f'Non-numeric distance value for pair ({i},{j}): {val_str}')
            d[i, j] = val
    for i in N:
        for j in N:
            if i == j:
                continue
            if (i, j) not in d:
                raise ValueError(f'Missing distance for pair ({i},{j})')
    V = N - {'Depot'}
    V_list = list(V)
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars([(i, j) for i in N for j in N if i != j], vtype=GRB.BINARY, name='')
    u = model.addVars([i for i in V], vtype=GRB.INTEGER, name='')
    model.setObjective(gurobipy.quicksum((d[i, j] * x[i, j] for i in N for j in N if i != j)), GRB.MINIMIZE)
    for i in N:
        model.addConstr(gurobipy.quicksum((x[i, j] for j in N if j != i)) == 1, name='')
    for j in N:
        model.addConstr(gurobipy.quicksum((x[i, j] for i in N if i != j)) == 1, name='')
    V_size = len(V)
    for i in V:
        for j in V:
            if i == j:
                continue
            model.addConstr(u[i] - u[j] + V_size * x[i, j] <= V_size - 1, name='')
    for i in V:
        model.addConstr(u[i] >= 1, name='')
        model.addConstr(u[i] <= V_size, name='')
    model.optimize()
    return model
m = solve_problem()