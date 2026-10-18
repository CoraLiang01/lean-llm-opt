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
 'route': 'TP',
 'tables': [{'aliases': ['file_0_view_0', 'parameters.csv'],
             'columns': ['truck_id', 'Q', 'S', 'C', 'd1', 'd2', 'd3', 'd4'],
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
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
data = CSVQA_DATA['tables'][0]['records']
trucks = []
Q = {}
S = {}
C = {}
demand = [0, 0, 0, 0]
for rec in data:
    vals = rec['values']
    tid = int(vals['truck_id'])
    trucks.append(tid)
    Q[tid] = float(vals['Q'])
    S[tid] = float(vals['S'])
    C[tid] = float(vals['C'])
    if demand[0] == 0:
        demand[1] = float(vals['d1'])
        demand[2] = float(vals['d2'])
        demand[3] = float(vals['d3'])
        demand[4] = float(vals['d4'])
trucks.sort()
m = gp.Model('truck_schedule')
m.Params.MIPGap = 1e-06
y = {}
s = {}
w = {}
for i in trucks:
    for t in range(1, 5):
        y[i, t] = m.addVar(vtype=GRB.BINARY)
        w[i, t] = m.addVar(vtype=GRB.CONTINUOUS, lb=0)
    for t in range(1, 4):
        s[i, t] = m.addVar(vtype=GRB.BINARY)
m.update()
for i in trucks:
    m.addConstr(s[i, 1] <= y[i, 1])
    m.addConstr(s[i, 2] <= y[i, 2])
    m.addConstr(s[i, 3] <= y[i, 3])
    m.addConstr(s[i, 2] >= y[i, 2] - y[i, 1])
    m.addConstr(s[i, 3] >= y[i, 3] - y[i, 2])
    m.addConstr(s[i, 1] >= y[i, 1])
for i in trucks:
    m.addConstr(y[i, 1] <= y[i, 2])
    m.addConstr(y[i, 2] <= y[i, 3])
    m.addConstr(y[i, 3] <= y[i, 4])
for i in trucks:
    m.addConstr(y[i, 4] <= y[i, 3])
for i in trucks:
    for t in range(1, 5):
        m.addConstr(w[i, t] <= Q[i] * y[i, t])
for i in trucks:
    m.addConstr(w[i, 1] <= 300)
    m.addConstr(w[i, 1] >= 0)
    for t in range(2, 5):
        m.addConstr(w[i, t] - w[i, t - 1] <= 300)
        m.addConstr(w[i, t - 1] - w[i, t] <= 300)
for t in range(1, 5):
    m.addConstr(gp.quicksum((w[i, t] for i in trucks)) >= demand[t])
for t in range(1, 5):
    m.addConstr(gp.quicksum((w[i, t] for i in trucks)) <= 0.9 * gp.quicksum((Q[i] * y[i, t] for i in trucks)))
obj = gp.quicksum((S[i] * (s[i, 1] + s[i, 2] + s[i, 3]) + gp.quicksum((C[i] * w[i, t] for t in range(1, 5))) for i in trucks))
m.setObjective(obj, GRB.MINIMIZE)
m.optimize()
if m.Status == GRB.OPTIMAL:
    print('Objective:', m.ObjVal)
    for v in m.getVars():
        print(v.VarName, v.X)
else:
    print('Solver status:', m.Status)