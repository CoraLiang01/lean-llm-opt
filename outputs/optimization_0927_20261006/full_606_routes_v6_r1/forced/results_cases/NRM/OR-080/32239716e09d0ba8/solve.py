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
 'route': 'NRM',
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
             'role': 'truck parameters and period demands',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB
table = None
for t in CSVQA_DATA['tables']:
    if t['table_id'] == 'file_0_view_0':
        table = t
        break
if table is None:
    raise ValueError("Required table_id 'file_0_view_0' not found in CSVQA_DATA.")
records = table['records']
if len(records) != 10:
    raise ValueError('Expected 10 trucks in parameters.csv.')
T = []
Q = {}
S = {}
C = {}
for rec in records:
    truck_id = rec['values']['truck_id']
    T.append(truck_id)
    Q[truck_id] = float(rec['values']['Q'])
    S[truck_id] = float(rec['values']['S'])
    C[truck_id] = float(rec['values']['C'])
P = [1, 2, 3, 4]
demand_row = records[0]['values']
d_p = {1: float(demand_row['d1']), 2: float(demand_row['d2']), 3: float(demand_row['d3']), 4: float(demand_row['d4'])}
m = gp.Model('Truck_Scheduling')
y_vars = m.addVars(T, P, vtype=GRB.BINARY, name='')
z_vars = m.addVars(T, P, vtype=GRB.BINARY, name='')
x_vars = m.addVars(T, P, lb=0.0, vtype=GRB.CONTINUOUS, name='')
x0 = {t: 0.0 for t in T}
y0 = {t: 0 for t in T}
m.setObjective(gp.quicksum((S[t] * z_vars[t, p] + C[t] * x_vars[t, p] for t in T for p in P)), GRB.MINIMIZE)
for t in T:
    for p in P:
        prev_p = p - 1
        y_prev = y0[t] if prev_p == 0 else y_vars[t, prev_p]
        m.addConstr(z_vars[t, p] >= y_vars[t, p] - y_prev)
        m.addConstr(z_vars[t, p] <= 1 - y_prev)
        m.addConstr(z_vars[t, p] <= y_vars[t, p])
    m.addConstr(z_vars[t, 4] == 0)
for t in T:
    for p in [1, 2, 3]:
        prev_p = p - 1
        y_prev = y0[t] if prev_p == 0 else y_vars[t, prev_p]
        m.addConstr(y_vars[t, p + 1] >= y_vars[t, p] - y_prev)
for t in T:
    for p in [1, 2, 3]:
        prev_p = p - 1
        y_prev = y0[t] if prev_p == 0 else y_vars[t, prev_p]
        m.addConstr(y_vars[t, p + 1] <= y_vars[t, p] + y_prev)
for t in T:
    for p in [1, 2]:
        prev_p = p - 1
        y_prev = y0[t] if prev_p == 0 else y_vars[t, prev_p]
        m.addConstr(y_vars[t, p] + y_prev >= y_vars[t, p + 1])
for t in T:
    for p in P:
        m.addConstr(x_vars[t, p] <= Q[t] * y_vars[t, p])
for t in T:
    for p in P:
        m.addConstr(x_vars[t, p] <= Q[t])
for t in T:
    for p in P:
        prev_p = p - 1
        x_prev = x0[t] if prev_p == 0 else x_vars[t, prev_p]
        m.addConstr(x_vars[t, p] - x_prev <= 300)
        m.addConstr(x_prev - x_vars[t, p] <= 300)
for p in P:
    m.addConstr(gp.quicksum((x_vars[t, p] for t in T)) >= d_p[p])
for p in P:
    m.addConstr(gp.quicksum((x_vars[t, p] for t in T)) <= 0.9 * gp.quicksum((Q[t] * y_vars[t, p] for t in T)))
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')