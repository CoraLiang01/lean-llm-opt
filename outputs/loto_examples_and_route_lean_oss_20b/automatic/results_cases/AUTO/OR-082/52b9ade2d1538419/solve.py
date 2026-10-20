CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A small courier company operates a single delivery van that must depart from the depot, visit three '
          'customer locations — A, B, and C — exactly once each in any order, and then return to the depot on the same '
          'day. The pairwise road distances (in kilometres) between the depot and every location are provided in '
          'DistanceMatrix.csv. There are no service-time or time-window constraints. Formulate this problem and '
          'determine the sequence of visits that minimises the total travel distance.',
 'relationships': [],
 'route': 'TP',
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
import pandas as pd
table = next((t for t in CSVQA_DATA['tables'] if t['table_id'] == 'file_0_view_0'))
records = table['records']
cols = table['columns']
df = pd.DataFrame([r['values'] for r in records], columns=cols)
df = df.set_index('Depot')
dist = df.astype(float).to_dict(orient='index')
nodes = ['Depot', 'A', 'B', 'C']
customers = ['A', 'B', 'C']
m = gp.Model()
m.setParam(GRB.Param.MIPGap, 1e-06)
x = {}
for i in nodes:
    for j in nodes:
        if i != j:
            x[i, j] = m.addVar(vtype=GRB.BINARY)
u = {}
for i in customers:
    u[i] = m.addVar(vtype=GRB.CONTINUOUS, lb=1, ub=3)
m.update()
obj = gp.quicksum((dist[i][j] * x[i, j] for i in nodes for j in nodes if i != j))
m.setObjective(obj, GRB.MINIMIZE)
m.addConstr(gp.quicksum((x['Depot', j] for j in customers)) == 1)
m.addConstr(gp.quicksum((x[i, 'Depot'] for i in customers)) == 1)
for i in customers:
    m.addConstr(gp.quicksum((x[i, j] for j in nodes if j != i)) == 1)
for i in customers:
    m.addConstr(gp.quicksum((x[j, i] for j in nodes if j != i)) == 1)
for i in customers:
    for j in customers:
        if i != j:
            m.addConstr(u[i] - u[j] + 3 * x[i, j] <= 2)
m.update()
m.optimize()
if m.status == GRB.OPTIMAL:
    print('ObjVal:', m.objVal)
    for v in m.getVars():
        print(v.varName, v.x)
else:
    print('Status:', m.status)