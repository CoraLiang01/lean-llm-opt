CSVQA_DATA = {'query': 'A small courier company operates a single delivery van that must depart from the depot, visit three customer locations — A, B, and C — exactly once each in any order, and then return to the depot on the same day. The pairwise road distances (in kilometres) between the depot and every location are provided in DistanceMatrix.csv. There are no service-time or time-window constraints. Formulate this problem and determine the sequence of visits that minimises the total travel distance.', 'relationships': [], 'tables': [{'columns': ['Unnamed: 0', 'Depot', 'A', 'B', 'C'], 'file_name': 'DistanceMatrix.csv', 'records': [{'data_row_number': 0, 'values': {'A': '28', 'B': '41', 'C': '63', 'Depot': '0', 'Unnamed: 0': 'Depot'}}, {'data_row_number': 1, 'values': {'A': '0', 'B': '27', 'C': '87', 'Depot': '28', 'Unnamed: 0': 'A'}}, {'data_row_number': 2, 'values': {'A': '27', 'B': '0', 'C': '81', 'Depot': '41', 'Unnamed: 0': 'B'}}, {'data_row_number': 3, 'values': {'A': '87', 'B': '81', 'C': '0', 'Depot': '63', 'Unnamed: 0': 'C'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table = next((t for t in CSVQA_DATA['tables'] if t.get('table_id') == 'file_0_view_0'))
    columns = table['columns']
    N = [col for col in columns if col != 'Unnamed: 0']
    d = {}
    for record in table['records']:
        row_label = record['values']['Unnamed: 0']
        for col in N:
            value_str = record['values'][col]
            try:
                value = float(value_str)
            except ValueError:
                raise ValueError(f'Non-numeric distance value: {value_str}')
            d[row_label, col] = value
    for i in N:
        for j in N:
            if (i, j) not in d:
                raise ValueError(f'Missing distance for pair ({i},{j})')
    m = gurobipy.Model('TSP_MTZ')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(i, j) for i in N for j in N], vtype=GRB.BINARY, name='')
    u = m.addVars([i for i in N if i != 'Depot'], vtype=GRB.CONTINUOUS, lb=1, ub=3, name='')
    m.setObjective(gurobipy.quicksum((d[i, j] * x[i, j] for i in N for j in N)), GRB.MINIMIZE)
    for i in N:
        m.addConstr(gurobipy.quicksum((x[i, j] for j in N)) == 1, name=f'out_{i}')
    for j in N:
        m.addConstr(gurobipy.quicksum((x[i, j] for i in N)) == 1, name=f'in_{j}')
    for i in N:
        if i == 'Depot':
            continue
        for j in N:
            if j == 'Depot' or i == j:
                continue
            m.addConstr(u[i] - u[j] + 3 * x[i, j] <= 2, name=f'sub_{i}_{j}')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')
