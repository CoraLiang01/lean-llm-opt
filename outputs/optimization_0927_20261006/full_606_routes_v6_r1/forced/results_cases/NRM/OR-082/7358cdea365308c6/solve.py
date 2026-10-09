CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A small courier company operates a single delivery van that must depart from the depot, visit three '
          'customer locations — A, B, and C — exactly once each in any order, and then return to the depot on the same '
          'day. The pairwise road distances (in kilometres) between the depot and every location are provided in '
          'DistanceMatrix.csv. There are no service-time or time-window constraints. Formulate this problem and '
          'determine the sequence of visits that minimises the total travel distance.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Unnamed: 0', 'Depot', 'A', 'B', 'C'],
             'file_index': 0,
             'file_name': 'DistanceMatrix.csv',
             'filters': {'conditions': [{'column': 'Unnamed: 0',
                                         'dtype': 'string',
                                         'evidence': 'visit three customer locations — A, B, and C — exactly once each '
                                                     'in any order, and then return to the depot',
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
             'role': 'distance matrix for routing',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB
distance_table = [r['values'] for r in CSVQA_DATA['tables'][0]['records']]
locations = [row['Unnamed: 0'] for row in distance_table]
N = locations
V = [loc for loc in N if loc != 'Depot']
d = {}
for row in distance_table:
    i = row['Unnamed: 0']
    for j in N:
        d[i, j] = float(row[j])
for i in N:
    for j in N:
        if i != j and (i, j) not in d:
            raise ValueError(f'Missing distance from {i} to {j}')
m = gp.Model('TSP_Courier')
x_vars = m.addVars([(i, j) for i in N for j in N if i != j], vtype=GRB.BINARY, name='')
u_vars = m.addVars(V, vtype=GRB.INTEGER, lb=1, ub=3, name='')
m.setObjective(gp.quicksum((d[i, j] * x_vars[i, j] for i in N for j in N if i != j)), GRB.MINIMIZE)
m.addConstr(gp.quicksum((x_vars['Depot', j] for j in N if j != 'Depot')) == 1, name='depart_depot')
m.addConstr(gp.quicksum((x_vars[i, 'Depot'] for i in N if i != 'Depot')) == 1, name='return_depot')
for k in V:
    m.addConstr(gp.quicksum((x_vars[i, k] for i in N if i != k)) == 1, name=f'visit_in_{k}')
    m.addConstr(gp.quicksum((x_vars[k, j] for j in N if j != k)) == 1, name=f'visit_out_{k}')
for i in V:
    for j in V:
        if i != j:
            m.addConstr(u_vars[i] - u_vars[j] + 3 * x_vars[i, j] <= 2, name=f'mtz_{i}_{j}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')