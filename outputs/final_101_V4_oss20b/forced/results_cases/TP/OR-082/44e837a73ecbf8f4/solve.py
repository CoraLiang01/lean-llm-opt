CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A small courier company operates a single delivery van that must depart from the depot, visit three '
          'customer locations — A, B, and C — exactly once each in any order, and then return to the depot on the same '
          'day. The pairwise road distances (in kilometres) between the depot and every location are provided in '
          'DistanceMatrix.csv. There are no service-time or time-window constraints. Formulate this problem and '
          'determine the sequence of visits that minimises the total travel distance.',
 'relationships': [],
 'route': 'TP',
 'tables': [{'aliases': ['file_0_view_0', 'DistanceMatrix.csv'],
             'columns': ['Unnamed: 0', 'Depot', 'A', 'B', 'C'],
             'file_index': 0,
             'file_name': 'DistanceMatrix.csv',
             'filters': {'conditions': [{'column': 'Unnamed: 0',
                                         'dtype': 'string',
                                         'evidence': 'depot, A, B, C',
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
import gurobipy as gp
from gurobipy import GRB
records = CSVQA_DATA['tables'][0]['records']
nodes = [rec['values']['Unnamed: 0'] for rec in records]
distances = {}
for rec in records:
    row = rec['values']['Unnamed: 0']
    distances[row] = {}
    for col in ['Depot', 'A', 'B', 'C']:
        distances[row][col] = int(rec['values'][col])
m = gp.Model()
m.setParam('MIPGap', 1e-06)
y = {}
for i in nodes:
    for j in nodes:
        if i != j:
            y[i, j] = m.addVar(vtype=GRB.BINARY)
u = {}
customers = [n for n in nodes if n != 'Depot']
for k in customers:
    u[k] = m.addVar(vtype=GRB.CONTINUOUS, lb=1, ub=3)
m.setObjective(gp.quicksum((distances[i][j] * y[i, j] for i in nodes for j in nodes if i != j)), GRB.MINIMIZE)
for i in nodes:
    m.addConstr(gp.quicksum((y[i, j] for j in nodes if j != i)) == 1)
for j in nodes:
    m.addConstr(gp.quicksum((y[i, j] for i in nodes if i != j)) == 1)
for i in customers:
    for j in customers:
        if i != j:
            m.addConstr(u[i] - u[j] + 3 * y[i, j] <= 2)
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(v.VarName, v.X)
else:
    print(f'Status: {m.Status}')