CSVQA_DATA = {'query': 'A small courier company operates a single delivery van that must depart from the depot, visit three customer locations — A, B, and C — exactly once each in any order, and then return to the depot on the same day. The pairwise road distances (in kilometres) between the depot and every location are provided in DistanceMatrix.csv. There are no service-time or time-window constraints. Formulate this problem and determine the sequence of visits that minimises the total travel distance.', 'relationships': [{'column_axis': {'id_column': 'Unnamed: 0', 'table_id': 'file_0_view_0'}, 'matrix_table_id': 'file_0_view_0', 'row_axis': {'id_column': 'Unnamed: 0', 'table_id': 'file_0_view_0'}, 'row_id_column': 'Unnamed: 0', 'type': 'matrix'}], 'tables': [{'columns': ['Depot', 'A', 'B', 'C', 'Unnamed: 0'], 'file_name': 'DistanceMatrix.csv', 'records': [{'data_row_number': 0, 'values': {'A': '28', 'B': '41', 'C': '63', 'Depot': '0', 'Unnamed: 0': 'Depot'}}, {'data_row_number': 1, 'values': {'A': '0', 'B': '27', 'C': '87', 'Depot': '28', 'Unnamed: 0': 'A'}}, {'data_row_number': 2, 'values': {'A': '27', 'B': '0', 'C': '81', 'Depot': '41', 'Unnamed: 0': 'B'}}, {'data_row_number': 3, 'values': {'A': '87', 'B': '81', 'C': '0', 'Depot': '63', 'Unnamed: 0': 'C'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table_id = 'file_0_view_0'
    table = next((t for t in CSVQA_DATA['tables'] if t['table_id'] == table_id))
    records = table['records']
    columns = table['columns']
    N = set()
    for rec in records:
        node = rec['values']['Unnamed: 0']
        N.add(node)
    C = N - {'Depot'}
    d = {}
    for rec in records:
        i = rec['values']['Unnamed: 0']
        for col in columns:
            if col == 'Unnamed: 0':
                continue
            j = col
            if i == j:
                continue
            val_str = rec['values'][col]
            d[i, j] = float(val_str)
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(i, j) for i in N for j in N if i != j], vtype=GRB.BINARY, name='')
    u = m.addVars([(i,) for i in C], vtype=GRB.INTEGER, lb=1, ub=len(C), name='')
    m.setObjective(gurobipy.quicksum((d[i, j] * x[i, j] for i in N for j in N if i != j)), GRB.MINIMIZE)
    m.addConstr(gurobipy.quicksum((x['Depot', j] for j in C)) == 1, name='depot_out')
    m.addConstr(gurobipy.quicksum((x[i, 'Depot'] for i in C)) == 1, name='depot_in')
    for i in C:
        m.addConstr(gurobipy.quicksum((x[i, j] for j in N if j != i)) == 1, name=f'cust_out_{i}')
    for i in C:
        m.addConstr(gurobipy.quicksum((x[j, i] for j in N if j != i)) == 1, name=f'cust_in_{i}')
    for i in C:
        for j in C:
            if i == j:
                continue
            m.addConstr(u[i] - u[j] + len(C) * x[i, j] <= len(C) - 1, name=f'mtz_{i}_{j}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()
