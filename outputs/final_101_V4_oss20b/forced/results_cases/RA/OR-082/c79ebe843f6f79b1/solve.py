CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A small courier company operates a single delivery van that must depart from the depot, visit three '
          'customer locations — A, B, and C — exactly once each in any order, and then return to the depot on the same '
          'day. The pairwise road distances (in kilometres) between the depot and every location are provided in '
          'DistanceMatrix.csv. There are no service-time or time-window constraints. Formulate this problem and '
          'determine the sequence of visits that minimises the total travel distance.',
 'relationships': [],
 'route': 'RA',
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
records = CSVQA_DATA['tables'][0]['records']
nodes = ['Depot', 'A', 'B', 'C']
dist = {}
for rec in records:
    row_node = None
    for node in nodes:
        if rec['values'][node] == '0':
            row_node = node
            break
    for node in nodes:
        if node != row_node:
            dist[row_node, node] = int(rec['values'][node])
m = gp.Model('TSP')
x = {}
for i in nodes:
    for j in nodes:
        if i != j:
            x[i, j] = m.addVar(vtype=GRB.BINARY)
u = {}
for i in nodes:
    if i != 'Depot':
        u[i] = m.addVar(vtype=GRB.CONTINUOUS, lb=0)
for i in nodes:
    m.addConstr(gp.quicksum((x[i, j] for j in nodes if j != i)) == 1)
for j in nodes:
    m.addConstr(gp.quicksum((x[i, j] for i in nodes if i != j)) == 1)
for i in nodes:
    if i == 'Depot':
        continue
    for j in nodes:
        if j == 'Depot' or i == j:
            continue
        m.addConstr(u[i] - u[j] + 3 * x[i, j] <= 2)
m.setObjective(gp.quicksum((dist[i, j] * x[i, j] for i in nodes for j in nodes if i != j)), GRB.MINIMIZE)
m.setParam('MIPGap', 1e-06)
m.optimize()
if m.status == GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} = {v.X}')
else:
    print(f'Status: {m.status}')