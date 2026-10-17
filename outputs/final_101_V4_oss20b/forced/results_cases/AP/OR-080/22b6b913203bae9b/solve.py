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
 'route': 'AP',
 'tables': [{'columns': ['truck_id', 'Q', 'S', 'C', 'd1', 'd2', 'd3', 'd4'],
             'file_index': 0,
             'file_name': 'parameters.csv',
             'filters': {'conditions': [], 'logic': 'and'},
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
    records = CSVQA_DATA['tables'][0]['records']
    I = [rec['values']['truck_id'] for rec in records]
    T = [1, 2, 3, 4]
    Q = {}
    S = {}
    C = {}
    for rec in records:
        tid = rec['values']['truck_id']
        Q[tid] = float(rec['values']['Q'])
        S[tid] = float(rec['values']['S'])
        C[tid] = float(rec['values']['C'])
    d = {1: float(records[0]['values']['d1']), 2: float(records[0]['values']['d2']), 3: float(records[0]['values']['d3']), 4: float(records[0]['values']['d4'])}
    for rec in records[1:]:
        if float(rec['values']['d1']) != d[1] or float(rec['values']['d2']) != d[2] or float(rec['values']['d3']) != d[3] or (float(rec['values']['d4']) != d[4]):
            raise ValueError('Demand values inconsistent across records')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    y = m.addVars([(i, t) for i in I for t in T], vtype=GRB.BINARY, name='')
    x = m.addVars([(i, t) for i in I for t in T], vtype=GRB.BINARY, name='')
    z = m.addVars([(i, t) for i in I for t in T], vtype=GRB.BINARY, name='')
    w = m.addVars([(i, t) for i in I for t in T], vtype=GRB.CONTINUOUS, lb=0, name='')
    obj = gurobipy.quicksum((S[i] * x[i, t] + C[i] * w[i, t] for i in I for t in T))
    m.setObjective(obj, GRB.MINIMIZE)
    m.addConstrs((y[i, 1] == x[i, 1] for i in I), name='')
    m.addConstrs((y[i, t] == y[i, t - 1] + x[i, t] - z[i, t] for i in I for t in [2, 3, 4]), name='')
    m.addConstrs((x[i, t] <= y[i, t] for i in I for t in T), name='')
    m.addConstrs((x[i, t] <= 1 - y[i, t - 1] for i in I for t in [2, 3, 4]), name='')
    m.addConstrs((x[i, 4] == 0 for i in I), name='')
    m.addConstrs((z[i, t] <= 1 - y[i, t] for i in I for t in T), name='')
    m.addConstrs((z[i, t] <= y[i, t - 1] for i in I for t in [2, 3, 4]), name='')
    m.addConstrs((y[i, t + 1] <= 1 - z[i, t] for i in I for t in [1, 2]), name='')
    m.addConstrs((y[i, t + 2] <= 1 - z[i, t] for i in I for t in [1]), name='')
    m.addConstrs((x[i, t + 2] <= 1 - z[i, t] for i in I for t in [1]), name='')
    m.addConstrs((y[i, t + 1] >= x[i, t] for i in I for t in [1, 2, 3]), name='')
    m.addConstrs((w[i, t] <= Q[i] * y[i, t] for i in I for t in T), name='')
    m.addConstrs((w[i, 1] <= 300 for i in I), name='')
    m.addConstrs((w[i, t] - w[i, t - 1] <= 300 for i in I for t in [2, 3, 4]), name='')
    m.addConstrs((w[i, t - 1] - w[i, t] <= 300 for i in I for t in [2, 3, 4]), name='')
    m.addConstrs((gurobipy.quicksum((w[i, t] for i in I)) >= d[t] for t in T), name='')
    m.addConstrs((gurobipy.quicksum((w[i, t] for i in I)) <= 0.9 * gurobipy.quicksum((Q[i] * y[i, t] for i in I)) for t in T), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()