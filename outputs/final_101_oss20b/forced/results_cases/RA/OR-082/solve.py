CSVQA_DATA = {'query': 'A small courier company operates a single delivery van that must depart from the depot, visit three customer locations — A, B, and C — exactly once each in any order, and then return to the depot on the same day. The pairwise road distances (in kilometres) between the depot and every location are provided in DistanceMatrix.csv. There are no service-time or time-window constraints. Formulate this problem and determine the sequence of visits that minimises the total travel distance.', 'relationships': [], 'tables': [{'columns': ['Unnamed: 0', 'Depot', 'A', 'B', 'C'], 'file_name': 'DistanceMatrix.csv', 'records': [{'data_row_number': 0, 'values': {'A': '28', 'B': '41', 'C': '63', 'Depot': '0', 'Unnamed: 0': 'Depot'}}, {'data_row_number': 1, 'values': {'A': '0', 'B': '27', 'C': '87', 'Depot': '28', 'Unnamed: 0': 'A'}}, {'data_row_number': 2, 'values': {'A': '27', 'B': '0', 'C': '81', 'Depot': '41', 'Unnamed: 0': 'B'}}, {'data_row_number': 3, 'values': {'A': '87', 'B': '81', 'C': '0', 'Depot': '63', 'Unnamed: 0': 'C'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table = None
    for tbl in CSVQA_DATA['tables']:
        if tbl.get('table_id') == 'file_0_view_0':
            table = tbl
            break
    if table is None:
        raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA")
    row_map = {}
    for rec in table['records']:
        row_label = rec['values']['Unnamed: 0']
        row_map[row_label] = rec['values']
    N = {'Depot', 'A', 'B', 'C'}
    for node in N:
        if node not in row_map:
            raise ValueError(f"Missing row for node '{node}' in distance matrix")
    d = {}
    for i in N:
        for j in N:
            if i == j:
                continue
            if j not in row_map[i]:
                raise ValueError(f"Missing distance from '{i}' to '{j}'")
            try:
                d[i, j] = float(row_map[i][j])
            except ValueError as e:
                raise ValueError(f"Invalid distance value from '{i}' to '{j}': {row_map[i][j]}") from e
    A_set = [(i, j) for i in N for j in N if i != j]
    S_set = [(i, j) for i in {'A', 'B', 'C'} for j in {'A', 'B', 'C'} if i != j]
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(A_set, vtype=GRB.BINARY, name='')
    u = model.addVars(['A', 'B', 'C'], vtype=GRB.INTEGER, lb=2, ub=3, name='')
    model.setObjective(gurobipy.quicksum((d[i, j] * x[i, j] for i, j in A_set)), GRB.MINIMIZE)
    model.addConstrs((gurobipy.quicksum((x[i, j] for j in N if j != i)) == 1 for i in N), name='')
    model.addConstrs((gurobipy.quicksum((x[i, j] for i in N if i != j)) == 1 for j in N), name='')
    model.addConstrs((u[i] - u[j] + 3 * x[i, j] <= 2 for i, j in S_set), name='')
    model.optimize()
    return model
m = solve_problem()
