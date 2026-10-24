CSVQA_DATA = {'query': "A logistics company has 10 candidate trucks available to transport goods over four consecutive time periods. The customer demands in periods 1 through 4 are 1,500 kg, 2,000 kg, 1,800 kg, and 1,000 kg. Each truck's maximum capacity, startup cost, and unit transportation cost are provided in parameters.csv. All trucks are initially off immediately before period 1, and activating a truck in period 1 or any later period incurs its startup cost.\n\n    Once a truck is started, it must remain active for at least two consecutive periods, so a truck may not be started in period 4. If a truck is shut down in period t after being active in period t-1, it must remain inactive in both period t and period t+1 and cannot be restarted before period t+2. An inactive truck must transport zero weight, while the weight transported by an active truck cannot exceed its maximum capacity.\n\n    For each truck, the change in transported weight between two adjacent periods, including transitions to or from zero load, cannot exceed 300 kg. In every period, the total transported weight must be at least the customer demand. A 10% spare-capacity buffer must also be maintained, so the total transported weight in each period cannot exceed 90% of the combined maximum capacity of the trucks active in that period.\n\n    Determine the truck activation, startup, and transported-weight schedule that minimizes total startup and transportation costs.", 'relationships': [], 'tables': [{'columns': ['truck_id', 'Q', 'S', 'C', 'd1', 'd2', 'd3', 'd4'], 'file_name': 'parameters.csv', 'records': [{'data_row_number': 0, 'values': {'C': '2.0', 'Q': '1000', 'S': '500', 'd1': '1500', 'd2': '2000', 'd3': '1800', 'd4': '1000', 'truck_id': '1'}}, {'data_row_number': 1, 'values': {'C': '3.0', 'Q': '800', 'S': '300', 'd1': '1500', 'd2': '2000', 'd3': '1800', 'd4': '1000', 'truck_id': '2'}}, {'data_row_number': 2, 'values': {'C': '2.5', 'Q': '1200', 'S': '400', 'd1': '1500', 'd2': '2000', 'd3': '1800', 'd4': '1000', 'truck_id': '3'}}, {'data_row_number': 3, 'values': {'C': '3.0', 'Q': '600', 'S': '250', 'd1': '1500', 'd2': '2000', 'd3': '1800', 'd4': '1000', 'truck_id': '4'}}, {'data_row_number': 4, 'values': {'C': '2.2', 'Q': '900', 'S': '450', 'd1': '1500', 'd2': '2000', 'd3': '1800', 'd4': '1000', 'truck_id': '5'}}, {'data_row_number': 5, 'values': {'C': '2.8', 'Q': '700', 'S': '280', 'd1': '1500', 'd2': '2000', 'd3': '1800', 'd4': '1000', 'truck_id': '6'}}, {'data_row_number': 6, 'values': {'C': '2.4', 'Q': '1100', 'S': '420', 'd1': '1500', 'd2': '2000', 'd3': '1800', 'd4': '1000', 'truck_id': '7'}}, {'data_row_number': 7, 'values': {'C': '3.2', 'Q': '500', 'S': '200', 'd1': '1500', 'd2': '2000', 'd3': '1800', 'd4': '1000', 'truck_id': '8'}}, {'data_row_number': 8, 'values': {'C': '2.1', 'Q': '1000', 'S': '480', 'd1': '1500', 'd2': '2000', 'd3': '1800', 'd4': '1000', 'truck_id': '9'}}, {'data_row_number': 9, 'values': {'C': '2.9', 'Q': '650', 'S': '260', 'd1': '1500', 'd2': '2000', 'd3': '1800', 'd4': '1000', 'truck_id': '10'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    table = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    records = table['records']
    T = [rec['values']['truck_id'] for rec in records]
    P = [1, 2, 3, 4]
    Q = {}
    S = {}
    C = {}
    for rec in records:
        t = rec['values']['truck_id']
        Q[t] = float(rec['values']['Q'])
        S[t] = float(rec['values']['S'])
        C[t] = float(rec['values']['C'])
    first = records[0]['values']
    D = {1: float(first['d1']), 2: float(first['d2']), 3: float(first['d3']), 4: float(first['d4'])}
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    a = m.addVars([(t, p) for t in T for p in P], vtype=GRB.BINARY, name='')
    y = m.addVars([(t, p) for t in T for p in P], vtype=GRB.BINARY, name='')
    z = m.addVars([(t, p) for t in T for p in P], vtype=GRB.BINARY, name='')
    w = m.addVars([(t, p) for t in T for p in P], vtype=GRB.CONTINUOUS, lb=0, name='')
    obj = gurobipy.quicksum((C[t] * w[t, p] + S[t] * y[t, p] for t in T for p in P))
    m.setObjective(obj, GRB.MINIMIZE)
    for t in T:
        m.addConstr(a[t, 1] == y[t, 1])
        for p in [2, 3, 4]:
            m.addConstr(a[t, p] == a[t, p - 1] + y[t, p] - z[t, p])
    for t in T:
        m.addConstr(y[t, 4] == 0)
        for p in [1, 2, 3]:
            m.addConstr(y[t, p] <= a[t, p + 1])
        for p in [2, 3, 4]:
            m.addConstr(y[t, p] <= 1 - a[t, p - 1])
    for t in T:
        for p in [1, 2, 3, 4]:
            m.addConstr(z[t, p] <= 1 - a[t, p])
        for p in [1, 2, 3]:
            m.addConstr(z[t, p] <= 1 - a[t, p + 1])
        for p in [1, 2]:
            m.addConstr(y[t, p + 2] <= 1 - z[t, p])
    for t in T:
        for p in P:
            m.addConstr(w[t, p] <= Q[t] * a[t, p])
    for t in T:
        for p in [2, 3, 4]:
            m.addConstr(w[t, p] - w[t, p - 1] <= 300)
            m.addConstr(w[t, p - 1] - w[t, p] <= 300)
    for p in P:
        m.addConstr(gurobipy.quicksum((w[t, p] for t in T)) >= D[p])
    for p in P:
        m.addConstr(gurobipy.quicksum((w[t, p] for t in T)) <= 0.9 * gurobipy.quicksum((Q[t] * a[t, p] for t in T)))
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()
