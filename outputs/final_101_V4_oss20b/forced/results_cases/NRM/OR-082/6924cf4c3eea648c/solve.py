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
 'route': 'NRM',
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
    tables = CSVQA_DATA['tables']
    table = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    columns = table['columns']
    row_id_col = columns[0]
    node_labels = [c for c in columns if c != row_id_col]
    N = node_labels
    C = [n for n in N if n != 'Depot']
    d = {}
    row_labels = set()
    for record in table['records']:
        values = record['values']
        i = values[row_id_col]
        row_labels.add(i)
        for j in node_labels:
            if i == j:
                continue
            val_str = values[j]
            try:
                val = float(val_str)
            except ValueError:
                raise ValueError(f'Invalid numeric value for d_{i}_{j}: {val_str}')
            d[i, j] = val
    if set(row_labels) != set(N):
        missing_rows = set(N) - row_labels
        raise ValueError(f'Missing rows in distance matrix: {missing_rows}')
    if set(node_labels) != set(N):
        missing_cols = set(N) - set(node_labels)
        raise ValueError(f'Missing columns in distance matrix: {missing_cols}')
    for i in N:
        for j in N:
            if i == j:
                continue
            if (i, j) not in d:
                raise ValueError(f'Missing distance d_{i}_{j}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars([(i, j) for i in N for j in N if i != j], vtype=GRB.BINARY, name='')
    u = model.addVars(C, vtype=GRB.INTEGER, lb=1, ub=3, name='')
    model.setObjective(gurobipy.quicksum((d[i, j] * x[i, j] for i in N for j in N if i != j)), GRB.MINIMIZE)
    model.addConstr(gurobipy.quicksum((x['Depot', j] for j in N if j != 'Depot')) == 1, name='depot_depart')
    model.addConstr(gurobipy.quicksum((x[i, 'Depot'] for i in N if i != 'Depot')) == 1, name='depot_arrive')
    for i in C:
        model.addConstr(gurobipy.quicksum((x[i, j] for j in N if j != i)) == 1, name=f'out_{i}')
    for j in C:
        model.addConstr(gurobipy.quicksum((x[i, j] for i in N if i != j)) == 1, name=f'in_{j}')
    for i in C:
        for j in C:
            if i == j:
                continue
            model.addConstr(u[i] - u[j] + 3 * x[i, j] <= 2, name=f'mtz_{i}_{j}')
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(f'ObjVal: {model.ObjVal}')
        for var in model.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {model.Status}')
    return model
m = solve_problem()