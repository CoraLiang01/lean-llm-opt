CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A small courier company operates a single delivery van that must depart from the depot, visit three '
          'customer locations — A, B, and C — exactly once each in any order, and then return to the depot on the same '
          'day. The pairwise road distances (in kilometres) between the depot and every location are provided in '
          'DistanceMatrix.csv. There are no service-time or time-window constraints. Formulate this problem and '
          'determine the sequence of visits that minimises the total travel distance.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'aliases': ['file_0_view_0', 'DistanceMatrix.csv'],
             'columns': ['Depot', 'A', 'B', 'C'],
             'file_index': 0,
             'file_name': 'DistanceMatrix.csv',
             'filters': {'conditions': [{'column': 'Unnamed: 0',
                                         'dtype': 'string',
                                         'evidence': 'Depot, A, B, C',
                                         'operator': 'in',
                                         'value': ['Depot', 'A', 'B', 'C']}],
                         'logic': 'and'},
             'original_rows': 11,
             'records': [{'source_row': 0, 'values': {'A': '28', 'B': '41', 'C': '63', 'Depot': '0'}},
                         {'source_row': 1, 'values': {'A': '0', 'B': '27', 'C': '87', 'Depot': '28'}},
                         {'source_row': 2, 'values': {'A': '27', 'B': '0', 'C': '81', 'Depot': '41'}},
                         {'source_row': 3, 'values': {'A': '87', 'B': '81', 'C': '0', 'Depot': '63'}}],
             'returned_rows': 4,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB
table = None
for t in CSVQA_DATA['tables']:
    if t['table_id'] == 'file_0_view_0' or 'file_0_view_0' in t.get('aliases', []):
        table = t
        break
if table is None:
    raise ValueError('Distance matrix table not found')
records = table['records']
columns = table['columns']
name_to_index = {'Depot': 0, 'A': 1, 'B': 2, 'C': 3}
N = 4
d = [[0] * N for _ in range(N)]
for rec in records:
    i = rec['source_row']
    for col in columns:
        j = name_to_index[col]
        d[i][j] = float(rec['values'][col])
m = gp.Model('tsp')
m.setParam('MIPGap', 1e-06)
x = {}
for i in range(N):
    for j in range(N):
        if i != j:
            x[i, j] = m.addVar(vtype=GRB.BINARY)
u = {}
for i in range(1, N):
    u[i] = m.addVar(vtype=GRB.INTEGER)
for i in range(N):
    m.addConstr(gp.quicksum((x[i, j] for j in range(N) if i != j)) == 1)
for j in range(N):
    m.addConstr(gp.quicksum((x[i, j] for i in range(N) if i != j)) == 1)
for i in range(1, N):
    for j in range(1, N):
        if i != j:
            m.addConstr(u[i] - u[j] + 4 * x[i, j] <= 3)
for i in range(1, N):
    m.addConstr(u[i] >= 1)
    m.addConstr(u[i] <= 3)
m.setObjective(gp.quicksum((d[i][j] * x[i, j] for i in range(N) for j in range(N) if i != j)), GRB.MINIMIZE)
m.optimize()
if m.Status == GRB.OPTIMAL:
    print('ObjVal:', m.ObjVal)
    for var in m.getVars():
        print(f'{var.VarName} = {var.X}')
else:
    print('Status:', m.Status)