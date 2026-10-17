CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A small courier company operates a single delivery van that must depart from the depot, visit three '
          'customer locations — A, B, and C — exactly once each in any order, and then return to the depot on the same '
          'day. The pairwise road distances (in kilometres) between the depot and every location are provided in '
          'DistanceMatrix.csv. There are no service-time or time-window constraints. Formulate this problem and '
          'determine the sequence of visits that minimises the total travel distance.',
 'relationships': [{'column_axis': {'id_column': 'Unnamed: 0', 'table_id': 'file_0_view_0'},
                    'matrix_format': 'wide',
                    'matrix_metadata_columns': [],
                    'matrix_table_id': 'file_0_view_0',
                    'matrix_value_columns': ['Depot', 'A', 'B', 'C'],
                    'row_axis': {'id_column': 'Unnamed: 0', 'table_id': 'file_0_view_0'},
                    'row_id_column': 'Unnamed: 0',
                    'type': 'matrix'}],
 'route': 'NRM',
 'tables': [{'columns': ['Unnamed: 0', 'Depot', 'A', 'B', 'C'],
             'file_index': 0,
             'file_name': 'DistanceMatrix.csv',
             'filters': {'conditions': [{'column': 'Unnamed: 0',
                                         'dtype': 'string',
                                         'evidence': 'depot, visit three customer locations — A, B, and C',
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
             'role': 'distance matrix',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [{'column_ids_aligned': True,
                                   'column_order_matches': True,
                                   'column_permutation': [0, 1, 2, 3],
                                   'expected_shape': [4, 4],
                                   'format': 'wide',
                                   'matrix_table_id': 'file_0_view_0',
                                   'metadata_columns': [],
                                   'row_ids_aligned': True,
                                   'row_order_matches': True,
                                   'row_permutation': [0, 1, 2, 3],
                                   'shape': [4, 4],
                                   'value_columns': ['Depot', 'A', 'B', 'C']}],
                'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB
table = None
for t in CSVQA_DATA['tables']:
    if t['table_id'] == 'file_0_view_0':
        table = t
        break
if table is None:
    raise ValueError("Distance matrix table 'file_0_view_0' not found.")
nodes = []
for rec in table['records']:
    node = rec['values']['Unnamed: 0']
    nodes.append(node)
if set(nodes) != {'Depot', 'A', 'B', 'C'}:
    raise ValueError('Node set does not match expected identifiers.')
dists = {}
for rec in table['records']:
    i = rec['values']['Unnamed: 0']
    for j in ['Depot', 'A', 'B', 'C']:
        val = rec['values'][j]
        try:
            dists[i, j] = float(val)
        except Exception:
            raise ValueError(f'Invalid distance value for ({i},{j}): {val}')
E = [(i, j) for i in nodes for j in nodes if i != j]
non_depot_nodes = [n for n in nodes if n != 'Depot']
for i, j in E:
    if (i, j) not in dists:
        raise ValueError(f'Missing distance for ({i},{j})')
m = gp.Model('TSP_Courier')
x_vars = m.addVars(E, vtype=GRB.BINARY, lb=0, ub=1, name='')
u_vars = m.addVars(non_depot_nodes, vtype=GRB.INTEGER, lb=1, ub=3, name='')
m.setObjective(gp.quicksum((dists[i, j] * x_vars[i, j] for i, j in E)), GRB.MINIMIZE)
for i in nodes:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in nodes if j != i)) == 1, name=f'depart_{i}')
for j in nodes:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in nodes if i != j)) == 1, name=f'arrive_{j}')
for i in non_depot_nodes:
    for j in non_depot_nodes:
        if i != j:
            m.addConstr(u_vars[i] - u_vars[j] + 3 * x_vars[i, j] <= 2, name=f'mtz_{i}_{j}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')