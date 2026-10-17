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
    table = CSVQA_DATA['tables'][0]
    records = table['records']
    trucks = [rec['values']['truck_id'] for rec in records]
    periods = [1, 2, 3, 4]
    Q = {rec['values']['truck_id']: float(rec['values']['Q']) for rec in records}
    S = {rec['values']['truck_id']: float(rec['values']['S']) for rec in records}
    C = {rec['values']['truck_id']: float(rec['values']['C']) for rec in records}
    d = {t: float(records[0]['values'][f'd{t}']) for t in periods}
    Delta = 300.0
    beta = 0.9
    model = gurobipy.Model('truck_schedule')
    model.setParam('MIPGap', 1e-06)
    y = model.addVars([(i, t) for i in trucks for t in periods], vtype=GRB.BINARY, name='')
    w = model.addVars([(i, t) for i in trucks for t in periods], vtype=GRB.CONTINUOUS, lb=0, name='')
    s = model.addVars([(i, t) for i in trucks for t in periods], vtype=GRB.BINARY, name='')
    obj = gurobipy.quicksum((C[i] * w[i, t] + S[i] * s[i, t] for i in trucks for t in periods))
    model.setObjective(obj, GRB.MINIMIZE)
    for t in periods:
        model.addConstr(gurobipy.quicksum((w[i, t] for i in trucks)) >= d[t], name=f'demand_t{t}')
    for t in periods:
        model.addConstr(gurobipy.quicksum((w[i, t] for i in trucks)) <= beta * gurobipy.quicksum((Q[i] * y[i, t] for i in trucks)), name=f'spare_t{t}')
    for i in trucks:
        for t in periods:
            model.addConstr(w[i, t] <= Q[i] * y[i, t], name=f'cap_i{i}_t{t}')
    for i in trucks:
        model.addConstr(w[i, 1] <= Delta, name=f'delta_start_i{i}')
        for t in periods[1:]:
            model.addConstr(w[i, t] - w[i, t - 1] <= Delta, name=f'delta_up_i{i}_t{t}')
            model.addConstr(w[i, t - 1] - w[i, t] <= Delta, name=f'delta_down_i{i}_t{t}')
    for i in trucks:
        for t in periods:
            y_prev = 0 if t == 1 else y[i, t - 1]
            model.addConstr(s[i, t] >= y[i, t] - y_prev, name=f'start_def1_i{i}_t{t}')
            model.addConstr(s[i, t] <= y[i, t], name=f'start_def2_i{i}_t{t}')
            model.addConstr(s[i, t] <= 1 - y_prev, name=f'start_def3_i{i}_t{t}')
    for i in trucks:
        model.addConstr(s[i, 4] == 0, name=f'no_start_t4_i{i}')
    for i in trucks:
        for t in periods[:-1]:
            y_prev = 0 if t == 1 else y[i, t - 1]
            model.addConstr(y[i, t + 1] >= y[i, t] - y_prev, name=f'min_consec1_i{i}_t{t}')
    for i in trucks:
        for t in periods[:-1]:
            y_prev = 0 if t == 1 else y[i, t - 1]
            model.addConstr(y[i, t + 1] <= y[i, t] + (1 - y_prev), name=f'shutdown1_i{i}_t{t}')
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(f'ObjVal: {model.ObjVal}')
        for var in model.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {model.Status}')
    return model
m = solve_problem()