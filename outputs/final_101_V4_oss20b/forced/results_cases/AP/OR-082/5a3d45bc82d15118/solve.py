CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A small courier company operates a single delivery van that must depart from the depot, visit three '
          'customer locations — A, B, and C — exactly once each in any order, and then return to the depot on the same '
          'day. The pairwise road distances (in kilometres) between the depot and every location are provided in '
          'DistanceMatrix.csv. There are no service-time or time-window constraints. Formulate this problem and '
          'determine the sequence of visits that minimises the total travel distance.',
 'relationships': [],
 'route': 'AP',
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
table = next((t for t in CSVQA_DATA['tables'] if t['table_id'] == 'file_0_view_0'))
records = table['records']
nodes = ['Depot', 'A', 'B', 'C']
dist = {row: {} for row in nodes}
for rec in records:
    row = rec['values']['Unnamed: 0']
    for col in ['Depot', 'A', 'B', 'C']:
        dist[row][col] = int(rec['values'][col])
m = gp.Model('TSP')
m.setParam(GRB.Param.MIPGap, 1e-06)
x = m.addVars([(i, j) for i in nodes for j in nodes if i != j], vtype=GRB.BINARY)
u = m.addVars(nodes[1:], vtype=GRB.INTEGER)
m.setObjective(gp.quicksum((dist[i][j] * x[i, j] for i, j in x.keys())), GRB.MINIMIZE)
m.addConstr(gp.quicksum((x['Depot', j] for j in nodes if j != 'Depot')) == 1, 'depot_out')
m.addConstr(gp.quicksum((x[i, 'Depot'] for i in nodes if i != 'Depot')) == 1, 'depot_in')
for k in nodes[1:]:
    m.addConstr(gp.quicksum((x[k, j] for j in nodes if j != k)) == 1, f'out_{k}')
    m.addConstr(gp.quicksum((x[i, k] for i in nodes if i != k)) == 1, f'in_{k}')
for i in nodes[1:]:
    for j in nodes[1:]:
        if i != j:
            m.addConstr(u[i] - u[j] + 4 * x[i, j] <= 3, f'mtz_{i}_{j}')
for k in nodes[1:]:
    m.addConstr(u[k] >= 2, f'lb_{k}')
    m.addConstr(u[k] <= 4, f'ub_{k}')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print('Objective:', m.ObjVal)
    for v in m.getVars():
        print(v.VarName, v.X)
else:
    print('Status:', m.Status)