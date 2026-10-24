CSVQA_DATA = {'query': 'A small courier company operates a single delivery van that must depart from the depot, visit three customer locations — A, B, and C — exactly once each in any order, and then return to the depot on the same day. The pairwise road distances (in kilometres) between the depot and every location are provided in DistanceMatrix.csv. There are no service-time or time-window constraints. Formulate this problem and determine the sequence of visits that minimises the total travel distance.', 'relationships': [], 'tables': [{'columns': ['Unnamed: 0', 'Depot', 'A', 'B', 'C'], 'file_name': 'DistanceMatrix.csv', 'records': [{'data_row_number': 0, 'values': {'A': '28', 'B': '41', 'C': '63', 'Depot': '0', 'Unnamed: 0': 'Depot'}}, {'data_row_number': 1, 'values': {'A': '0', 'B': '27', 'C': '87', 'Depot': '28', 'Unnamed: 0': 'A'}}, {'data_row_number': 2, 'values': {'A': '27', 'B': '0', 'C': '81', 'Depot': '41', 'Unnamed: 0': 'B'}}, {'data_row_number': 3, 'values': {'A': '87', 'B': '81', 'C': '0', 'Depot': '63', 'Unnamed: 0': 'C'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    table = next((t for t in CSVQA_DATA['tables'] if t['table_id'] == 'file_0_view_0'))
    columns = table['columns']
    nodes = [rec['values']['Unnamed: 0'] for rec in table['records']]
    d = {}
    for rec in table['records']:
        i = rec['values']['Unnamed: 0']
        for j in columns:
            if j == 'Unnamed: 0':
                continue
            d[i, j] = float(rec['values'][j])
    N = set(nodes)
    C = N - {'Depot'}
    x = m.addVars([(i, j) for i in N for j in N if i != j], vtype=GRB.BINARY, name='')
    u = m.addVars(list(C), vtype=GRB.INTEGER, lb=1, ub=len(C), name='')
    m.setObjective(gurobipy.quicksum((d[i, j] * x[i, j] for i in N for j in N if i != j)), GRB.MINIMIZE)
    for i in N:
        m.addConstr(gurobipy.quicksum((x[i, j] for j in N if i != j)) == 1, name='')
    for j in N:
        m.addConstr(gurobipy.quicksum((x[i, j] for i in N if i != j)) == 1, name='')
    for i in C:
        for j in C:
            if i == j:
                continue
            m.addConstr(u[i] - u[j] + len(C) * x[i, j] <= len(C) - 1, name='')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print('Objective:', m.ObjVal)
    for var in m.getVars():
        print(f'{var.VarName} = {var.X}')
else:
    print('Solver status:', m.Status)
