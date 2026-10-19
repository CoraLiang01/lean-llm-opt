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
import gurobipy as gp
from gurobipy import GRB
data = CSVQA_DATA
table = next((t for t in data['tables'] if t['table_id'] == 'file_0_view_0'))
records = table['records']
K = [int(rec['values']['truck_id']) for rec in records]
T = [1, 2, 3, 4]
Q = {int(rec['values']['truck_id']): float(rec['values']['Q']) for rec in records}
S = {int(rec['values']['truck_id']): float(rec['values']['S']) for rec in records}
C = {int(rec['values']['truck_id']): float(rec['values']['C']) for rec in records}
d = {t: float(records[0]['values'][f'd{t}']) for t in T}
m = gp.Model()
m.setParam('MIPGap', 1e-06)
x = {(k, t): m.addVar(vtype=GRB.CONTINUOUS, lb=0) for k in K for t in T}
y = {(k, t): m.addVar(vtype=GRB.BINARY) for k in K for t in T}
z = {(k, t): m.addVar(vtype=GRB.BINARY) for k in K for t in T}
m.update()
obj = gp.quicksum((S[k] * z[k, t] + C[k] * x[k, t] for k in K for t in T))
m.setObjective(obj, GRB.MINIMIZE)
for k in K:
    m.addConstr(z[k, 1] == y[k, 1])
    for t in [2, 3, 4]:
        m.addConstr(z[k, t] >= y[k, t] - y[k, t - 1])
        m.addConstr(z[k, t] <= y[k, t])
        m.addConstr(z[k, t] <= 1 - y[k, t - 1])
    m.addConstr(z[k, 4] == 0)
for k in K:
    for t in [1, 2, 3]:
        m.addConstr(y[k, t] <= y[k, t + 1])
    m.addConstr(y[k, 4] <= y[k, 3])
for k in K:
    for t in [2, 3]:
        m.addConstr(y[k, t - 1] + y[k, t + 1] <= 1 + y[k, t])
for k in K:
    for t in T:
        m.addConstr(x[k, t] <= Q[k] * y[k, t])
for k in K:
    m.addConstr(x[k, 1] <= 300)
    for t in [2, 3, 4]:
        m.addConstr(x[k, t] - x[k, t - 1] <= 300)
        m.addConstr(x[k, t] - x[k, t - 1] >= -300)
for t in T:
    m.addConstr(gp.quicksum((x[k, t] for k in K)) >= d[t])
for t in T:
    m.addConstr(gp.quicksum((x[k, t] for k in K)) <= 0.9 * gp.quicksum((Q[k] * y[k, t] for k in K)))
m.optimize()
if m.status == GRB.OPTIMAL:
    print(m.ObjVal)
    for v in m.getVars():
        print(v.VarName, v.X)
else:
    print(m.status)