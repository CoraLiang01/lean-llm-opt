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
    table = None
    for tbl in CSVQA_DATA['tables']:
        if tbl.get('table_id') == 'file_0_view_0':
            table = tbl
            break
    if table is None:
        raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA")
    trucks = []
    Q = {}
    S = {}
    C = {}
    for rec in table['records']:
        vals = rec['values']
        t_id = vals['truck_id']
        trucks.append(t_id)
        Q[t_id] = float(vals['Q'])
        S[t_id] = float(vals['S'])
        C[t_id] = float(vals['C'])
    d = {}
    first_rec = table['records'][0]['values']
    for p in range(1, 5):
        d[p] = float(first_rec[f'd{p}'])
    for t in trucks:
        if t not in Q or t not in S or t not in C:
            raise ValueError(f'Missing parameters for truck {t}')
    periods = [1, 2, 3, 4]
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    y = m.addVars([(t, p) for t in trucks for p in periods], vtype=GRB.BINARY, name='')
    s = m.addVars([(t, p) for t in trucks for p in periods], vtype=GRB.BINARY, name='')
    z = m.addVars([(t, p) for t in trucks for p in periods], vtype=GRB.BINARY, name='')
    w = m.addVars([(t, p) for t in trucks for p in periods], vtype=GRB.CONTINUOUS, lb=0, name='')
    for t in trucks:
        for p in periods:
            prev_y = 0 if p == 1 else y[t, p - 1]
            m.addConstr(y[t, p] == prev_y + s[t, p] - z[t, p], name=f'dyn_{t}_{p}')
    for t in trucks:
        for p in periods[:-1]:
            m.addConstr(s[t, p] <= y[t, p + 1], name=f'minup_{t}_{p}')
        m.addConstr(s[t, 4] == 0, name=f'start4_{t}')
    for t in trucks:
        for p in periods[:-1]:
            m.addConstr(z[t, p] <= 1 - y[t, p + 1], name=f'mindown1_{t}_{p}')
        for p in periods[:-2]:
            m.addConstr(z[t, p] <= 1 - y[t, p + 2], name=f'mindown2_{t}_{p}')
    for t in trucks:
        for p in periods:
            m.addConstr(w[t, p] <= Q[t] * y[t, p], name=f'weightlim_{t}_{p}')
    for t in trucks:
        for p in periods:
            prev_w = 0 if p == 1 else w[t, p - 1]
            m.addConstr(w[t, p] - prev_w <= 300, name=f'rampup_{t}_{p}')
            m.addConstr(prev_w - w[t, p] <= 300, name=f'rampdown_{t}_{p}')
    for p in periods:
        m.addConstr(gurobipy.quicksum((w[t, p] for t in trucks)) >= d[p], name=f'demand_{p}')
    for p in periods:
        total_capacity = gurobipy.quicksum((Q[t] * y[t, p] for t in trucks))
        m.addConstr(gurobipy.quicksum((w[t, p] for t in trucks)) <= 0.9 * total_capacity, name=f'spare_{p}')
    obj = gurobipy.quicksum((S[t] * s[t, p] + C[t] * w[t, p] for t in trucks for p in periods))
    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()
    return m
m = solve_problem()