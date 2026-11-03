CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A small courier company operates a single delivery van that must depart from the depot, visit three '
          'customer locations — A, B, and C — exactly once each in any order, and then return to the depot on the same '
          'day. The pairwise road distances (in kilometres) between the depot and every location are provided in '
          'DistanceMatrix.csv. There are no service-time or time-window constraints. Formulate this problem and '
          'determine the sequence of visits that minimises the total travel distance.',
 'relationships': [],
 'route': 'Others',
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
    table = CSVQA_DATA['tables'][0]
    records = table['records']
    nodes = ['Depot', 'A', 'B', 'C']
    d = {}
    for rec in records:
        i = rec['values']['Unnamed: 0']
        for j in nodes:
            if i == j:
                continue
            val = rec['values'][j]
            d[i, j] = float(val)
    m = gurobipy.Model('TSP_MTZ')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(i, j) for i in nodes for j in nodes if i != j], vtype=GRB.BINARY, name='')
    u = m.addVars(['A', 'B', 'C'], lb=1, ub=3, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gurobipy.quicksum((d[i, j] * x[i, j] for i in nodes for j in nodes if i != j)), GRB.MINIMIZE)
    for i in nodes:
        m.addConstr(gurobipy.quicksum((x[i, j] for j in nodes if j != i)) == 1, name=f'out_{i}')
    for j in nodes:
        m.addConstr(gurobipy.quicksum((x[i, j] for i in nodes if i != j)) == 1, name=f'in_{j}')
    for i in ['A', 'B', 'C']:
        for j in ['A', 'B', 'C']:
            if i == j:
                continue
            m.addConstr(u[i] - u[j] + 3 * x[i, j] <= 2, name=f'mtz_{i}_{j}')
    m.optimize()
    return m
m = solve_problem()