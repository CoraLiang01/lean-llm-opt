CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A logistics company has 10 candidate trucks available to transport goods over four consecutive time '
          'periods. The customer demands in periods 1 through 4 are 1,500 kg, 2,000 kg, 1,800 kg, and 1,000 kg. Each '
          "truck's maximum capacity, startup cost, and unit transportation cost are provided in parameters.csv. All "
          'trucks are initially off immediately before period 1, and activating a truck in period 1 or any later '
          'period incurs its startup cost.\n'
          '\n'
          '    Once a truck is started, it must remain active for at least two consecutive periods, so a truck may not '
          'be started in period 4. If a truck is shut down in period t after being active in period t-1, it must '
          'remain inactive in both period t and period t+1 and cannot be restarted before period t+2. An inactive '
          'truck must transport zero weight, while the weight transported by an active truck cannot exceed its maximum '
          'capacity.\n'
          '\n'
          '    For each truck, the change in transported weight between two adjacent periods, including transitions to '
          'or from zero load, cannot exceed 300 kg. In every period, the total transported weight must be at least the '
          'customer demand. A 10% spare-capacity buffer must also be maintained, so the total transported weight in '
          'each period cannot exceed 90% of the combined maximum capacity of the trucks active in that period.\n'
          '\n'
          '    Determine the truck activation, startup, and transported-weight schedule that minimizes total startup '
          'and transportation costs.',
 'relationships': [],
 'route': 'Others',
 'tables': [{'columns': ['truck_id', 'Q', 'S', 'C', 'd1', 'd2', 'd3', 'd4'],
             'file_index': 0,
             'file_name': 'parameters.csv',
             'filters': {},
             'original_rows': 10,
             'records': [{'source_row': 0,
                          'values': {'C': '2.0',
                                     'Q': '1000',
                                     'S': '500',
                                     'd1': '1500',
                                     'd2': '2000',
                                     'd3': '1800',
                                     'd4': '1000',
                                     'truck_id': '1'}},
                         {'source_row': 1,
                          'values': {'C': '3.0',
                                     'Q': '800',
                                     'S': '300',
                                     'd1': '1500',
                                     'd2': '2000',
                                     'd3': '1800',
                                     'd4': '1000',
                                     'truck_id': '2'}},
                         {'source_row': 2,
                          'values': {'C': '2.5',
                                     'Q': '1200',
                                     'S': '400',
                                     'd1': '1500',
                                     'd2': '2000',
                                     'd3': '1800',
                                     'd4': '1000',
                                     'truck_id': '3'}},
                         {'source_row': 3,
                          'values': {'C': '3.0',
                                     'Q': '600',
                                     'S': '250',
                                     'd1': '1500',
                                     'd2': '2000',
                                     'd3': '1800',
                                     'd4': '1000',
                                     'truck_id': '4'}},
                         {'source_row': 4,
                          'values': {'C': '2.2',
                                     'Q': '900',
                                     'S': '450',
                                     'd1': '1500',
                                     'd2': '2000',
                                     'd3': '1800',
                                     'd4': '1000',
                                     'truck_id': '5'}},
                         {'source_row': 5,
                          'values': {'C': '2.8',
                                     'Q': '700',
                                     'S': '280',
                                     'd1': '1500',
                                     'd2': '2000',
                                     'd3': '1800',
                                     'd4': '1000',
                                     'truck_id': '6'}},
                         {'source_row': 6,
                          'values': {'C': '2.4',
                                     'Q': '1100',
                                     'S': '420',
                                     'd1': '1500',
                                     'd2': '2000',
                                     'd3': '1800',
                                     'd4': '1000',
                                     'truck_id': '7'}},
                         {'source_row': 7,
                          'values': {'C': '3.2',
                                     'Q': '500',
                                     'S': '200',
                                     'd1': '1500',
                                     'd2': '2000',
                                     'd3': '1800',
                                     'd4': '1000',
                                     'truck_id': '8'}},
                         {'source_row': 8,
                          'values': {'C': '2.1',
                                     'Q': '1000',
                                     'S': '480',
                                     'd1': '1500',
                                     'd2': '2000',
                                     'd3': '1800',
                                     'd4': '1000',
                                     'truck_id': '9'}},
                         {'source_row': 9,
                          'values': {'C': '2.9',
                                     'Q': '650',
                                     'S': '260',
                                     'd1': '1500',
                                     'd2': '2000',
                                     'd3': '1800',
                                     'd4': '1000',
                                     'truck_id': '10'}}],
             'returned_rows': 10,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table = next((t for t in CSVQA_DATA['tables'] if t['table_id'] == 'file_0_view_0'))
    records = table['records']
    I = set()
    Q = {}
    S = {}
    C = {}
    for rec in records:
        vals = rec['values']
        i = int(vals['truck_id'])
        I.add(i)
        Q[i] = float(vals['Q'])
        S[i] = float(vals['S'])
        C[i] = float(vals['C'])
    T = [1, 2, 3, 4]
    T0 = [0]
    first_rec = records[0]['values']
    d = {1: float(first_rec['d1']), 2: float(first_rec['d2']), 3: float(first_rec['d3']), 4: float(first_rec['d4'])}
    for i in I:
        if i not in Q or i not in S or i not in C:
            raise ValueError(f'Missing parameter for truck {i}')
    for t in T:
        if t not in d:
            raise ValueError(f'Missing demand for period {t}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    y = model.addVars([(i, 0)] + [(i, t) for t in T], vtype=GRB.BINARY, name='')
    s = model.addVars([(i, t) for t in T], vtype=GRB.BINARY, name='')
    z = model.addVars([(i, t) for t in [1, 2, 3]], vtype=GRB.BINARY, name='')
    w = model.addVars([(i, t) for t in [0, 1, 2, 3, 4]], vtype=GRB.CONTINUOUS, lb=0, name='')
    for i in I:
        model.addConstr(y[i, 0] == 0)
    for i in I:
        for t in T:
            model.addConstr(s[i, t] >= y[i, t] - y[i, t - 1])
            model.addConstr(s[i, t] <= y[i, t])
            model.addConstr(s[i, t] <= 1 - y[i, t - 1])
        model.addConstr(s[i, 4] == 0)
    for i in I:
        for t in [1, 2, 3]:
            model.addConstr(y[i, t + 1] >= y[i, t] - y[i, t - 1])
    for i in I:
        for t in [1, 2, 3]:
            model.addConstr(z[i, t] >= y[i, t - 1] - y[i, t])
            model.addConstr(z[i, t] <= y[i, t - 1])
            model.addConstr(z[i, t] <= 1 - y[i, t])
    for i in I:
        for t in [1, 2, 3]:
            model.addConstr(y[i, t + 1] <= 1 - z[i, t])
        for t in [1, 2]:
            model.addConstr(y[i, t + 2] <= 1 - z[i, t])
    for i in I:
        for t in T:
            model.addConstr(w[i, t] <= Q[i] * y[i, t])
    for i in I:
        for t in T:
            model.addConstr(w[i, t] - w[i, t - 1] <= 300)
            model.addConstr(w[i, t - 1] - w[i, t] <= 300)
        model.addConstr(w[i, 0] == 0)
    for t in T:
        model.addConstr(gurobipy.quicksum((w[i, t] for i in I)) >= d[t])
    for t in T:
        model.addConstr(gurobipy.quicksum((w[i, t] for i in I)) <= 0.9 * gurobipy.quicksum((Q[i] * y[i, t] for i in I)))
    obj = gurobipy.quicksum((S[i] * s[i, t] + C[i] * w[i, t] for i in I for t in T))
    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(model.ObjVal)
        for var in model.getVars():
            print(var.VarName, var.X)
    else:
        print(model.Status)
    return model
m = solve_problem()