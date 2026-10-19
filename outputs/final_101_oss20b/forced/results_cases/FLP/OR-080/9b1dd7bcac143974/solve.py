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
 'route': 'FLP',
 'tables': [{'aliases': ['file_0_view_0', 'parameters.csv'],
             'columns': ['truck_id', 'Q', 'S', 'C', 'd1', 'd2', 'd3', 'd4'],
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
import gurobipy as gp
from gurobipy import GRB
import numpy as np
table = next((t for t in CSVQA_DATA['tables'] if t['table_id'] == 'file_0_view_0'))
records = table['records']
n_trucks = len(records)
n_periods = 4
Q = np.array([float(r['values']['Q']) for r in records])
S = np.array([float(r['values']['S']) for r in records])
C = np.array([float(r['values']['C']) for r in records])
demand = np.array([float(records[0]['values']['d1']), float(records[0]['values']['d2']), float(records[0]['values']['d3']), float(records[0]['values']['d4'])])
m = gp.Model('Truck_Activation')
x = m.addVars(n_trucks, n_periods, lb=0)
y = m.addVars(n_trucks, n_periods, vtype=GRB.BINARY)
z = m.addVars(n_trucks, n_periods, vtype=GRB.BINARY)
obj = gp.quicksum((C[i] * x[i, t] for i in range(n_trucks) for t in range(n_periods))) + gp.quicksum((S[i] * z[i, t] for i in range(n_trucks) for t in range(n_periods)))
m.setObjective(obj, GRB.MINIMIZE)
for t in range(n_periods):
    m.addConstr(gp.quicksum((x[i, t] for i in range(n_trucks))) >= demand[t])
for t in range(n_periods):
    m.addConstr(gp.quicksum((x[i, t] for i in range(n_trucks))) <= 0.9 * gp.quicksum((Q[i] * y[i, t] for i in range(n_trucks))))
for i in range(n_trucks):
    for t in range(n_periods):
        m.addConstr(x[i, t] <= Q[i] * y[i, t])
for i in range(n_trucks):
    m.addConstr(x[i, 0] <= 300)
    for t in range(1, n_periods):
        m.addConstr(x[i, t] - x[i, t - 1] <= 300)
        m.addConstr(x[i, t - 1] - x[i, t] <= 300)
for i in range(n_trucks):
    for t in range(n_periods):
        m.addConstr(y[i, t] >= z[i, t])
        if t > 0:
            m.addConstr(y[i, t] <= y[i, t - 1] + z[i, t])
for i in range(n_trucks):
    for t in range(n_periods - 1):
        m.addConstr(y[i, t + 1] >= z[i, t])
    for t in range(n_periods - 2):
        m.addConstr(y[i, t + 2] >= z[i, t + 1])
for i in range(n_trucks):
    for t in range(n_periods - 1):
        m.addConstr(y[i, t + 1] <= y[i, t])
    for t in range(n_periods - 2):
        m.addConstr(y[i, t + 2] <= y[i, t])
for i in range(n_trucks):
    m.addConstr(z[i, 3] == 0)
m.Params.MIPGap = 1e-06
m.optimize()
if m.Status == GRB.OPTIMAL:
    print('ObjVal:', m.ObjVal)
    for v in m.getVars():
        print(v.VarName, v.X)
else:
    print('Status:', m.Status)