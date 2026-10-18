CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A small courier company operates a single delivery van that must depart from the depot, visit three '
          'customer locations — A, B, and C — exactly once each in any order, and then return to the depot on the same '
          'day. The pairwise road distances (in kilometres) between the depot and every location are provided in '
          'DistanceMatrix.csv. There are no service-time or time-window constraints. Formulate this problem and '
          'determine the sequence of visits that minimises the total travel distance.',
 'relationships': [],
 'route': 'FLP',
 'tables': [{'aliases': ['file_0_view_0', 'DistanceMatrix.csv'],
             'columns': ['Depot', 'A', 'B', 'C'],
             'file_index': 0,
             'file_name': 'DistanceMatrix.csv',
             'filters': {'conditions': [{'column': 'Unnamed: 0',
                                         'dtype': 'string',
                                         'evidence': 'depot, A, B, C',
                                         'inclusive': 'both',
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
import numpy as np
nodes = ['Depot', 'A', 'B', 'C']
n = len(nodes)
dist = np.zeros((n, n), dtype=float)
node_index = {node: idx for idx, node in enumerate(nodes)}
records = CSVQA_DATA['tables'][0]['records']
for rec in records:
    values = rec['values']
    i = node_index[values['Depot']]
    for j_name in nodes[1:]:
        j = node_index[j_name]
        dist[i, j] = float(values[j_name])
m = gp.Model('TSP')
m.setParam('MIPGap', 1e-06)
x = m.addVars(n, n, vtype=GRB.BINARY)
u = m.addVars(1, 3, vtype=GRB.CONTINUOUS)
m.setObjective(gp.quicksum((dist[i, j] * x[i, j] for i in range(n) for j in range(n) if i != j)), GRB.MINIMIZE)
for i in range(n):
    m.addConstr(gp.quicksum((x[i, j] for j in range(n) if i != j)) == 1)
for j in range(n):
    m.addConstr(gp.quicksum((x[i, j] for i in range(n) if i != j)) == 1)
for i in range(1, n):
    for j in range(1, n):
        if i != j:
            m.addConstr(u[i] - u[j] + (n - 1) * x[i, j] <= n - 2)
for i in range(1, n):
    m.addConstr(u[i] >= 1)
    m.addConstr(u[i] <= n - 1)
m.optimize()
if m.status == GRB.OPTIMAL:
    print('Objective:', m.objVal)
    for v in m.getVars():
        if v.X > 1e-06:
            print(v.VarName, v.X)
else:
    print('Solver status:', m.status)