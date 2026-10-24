CSVQA_DATA = {'query': 'A small courier company operates a single delivery van that must depart from the depot, visit three customer locations — A, B, and C — exactly once each in any order, and then return to the depot on the same day. The pairwise road distances (in kilometres) between the depot and every location are provided in DistanceMatrix.csv. There are no service-time or time-window constraints. Formulate this problem and determine the sequence of visits that minimises the total travel distance.', 'relationships': [], 'tables': [{'columns': ['Unnamed: 0', 'Depot', 'A', 'B', 'C'], 'file_name': 'DistanceMatrix.csv', 'records': [{'data_row_number': 0, 'values': {'A': '28', 'B': '41', 'C': '63', 'Depot': '0', 'Unnamed: 0': 'Depot'}}, {'data_row_number': 1, 'values': {'A': '0', 'B': '27', 'C': '87', 'Depot': '28', 'Unnamed: 0': 'A'}}, {'data_row_number': 2, 'values': {'A': '27', 'B': '0', 'C': '81', 'Depot': '41', 'Unnamed: 0': 'B'}}, {'data_row_number': 3, 'values': {'A': '87', 'B': '81', 'C': '0', 'Depot': '63', 'Unnamed: 0': 'C'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table = CSVQA_DATA['tables'][0]
    records = table['records']
    columns = table['columns']
    nodes = set()
    for rec in records:
        nodes.add(rec['values']['Unnamed: 0'])
    col_nodes = set(columns[1:])
    if nodes != col_nodes:
        raise ValueError('Node identifiers in rows and columns do not match.')
    distances = {}
    for rec in records:
        i = rec['values']['Unnamed: 0']
        for col in columns[1:]:
            j = col
            if i == j:
                continue
            val_str = rec['values'][col]
            try:
                val = float(val_str)
            except ValueError:
                raise ValueError(f'Invalid numeric value for distance {i}->{j}: {val_str}')
            distances[i, j] = val
    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            if (i, j) not in distances:
                raise ValueError(f'Missing distance for pair {i}->{j}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(i, j) for i in nodes for j in nodes if i != j], vtype=GRB.BINARY, name='')
    aux_nodes = ['A', 'B', 'C']
    u = m.addVars(aux_nodes, vtype=GRB.INTEGER, lb=1, ub=2, name='')
    m.setObjective(gurobipy.quicksum((distances[i, j] * x[i, j] for i in nodes for j in nodes if i != j)), GRB.MINIMIZE)
    for i in nodes:
        m.addConstr(gurobipy.quicksum((x[i, j] for j in nodes if j != i)) == 1, name=f'out_{i}')
    for j in nodes:
        m.addConstr(gurobipy.quicksum((x[i, j] for i in nodes if i != j)) == 1, name=f'in_{j}')
    for i in aux_nodes:
        for j in aux_nodes:
            if i == j:
                continue
            m.addConstr(u[i] - u[j] + 3 * x[i, j] <= 2, name=f'mtz_{i}_{j}')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')
