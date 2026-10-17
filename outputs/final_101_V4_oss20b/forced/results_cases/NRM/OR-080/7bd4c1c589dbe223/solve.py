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
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    table = next((t for t in tables if t.get('table_id') == 'file_0_view_0'), None)
    if table is None:
        raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA.")
    records = table['records']
    trucks = []
    Q_t = {}
    S_t = {}
    C_t = {}
    d_p = {}
    for rec in records:
        vals = rec['values']
        truck_id = vals['truck_id']
        trucks.append(truck_id)
        try:
            Q_t[truck_id] = float(vals['Q'])
            S_t[truck_id] = float(vals['S'])
            C_t[truck_id] = float(vals['C'])
        except KeyError as e:
            raise ValueError(f'Missing parameter in record {rec['source_row']}: {e}')
        if not d_p:
            try:
                d_p[1] = float(vals['d1'])
                d_p[2] = float(vals['d2'])
                d_p[3] = float(vals['d3'])
                d_p[4] = float(vals['d4'])
            except KeyError as e:
                raise ValueError(f'Missing demand parameter in record {rec['source_row']}: {e}')
    for t in trucks:
        if t not in Q_t or t not in S_t or t not in C_t:
            raise ValueError(f'Missing parameters for truck {t}')
    if len(d_p) != 4:
        raise ValueError('Demand parameters for all 4 periods are required.')
    periods = [1, 2, 3, 4]
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    a = m.addVars([(t, p) for t in trucks for p in periods], vtype=GRB.BINARY, name='')
    s = m.addVars([(t, p) for t in trucks for p in periods], vtype=GRB.BINARY, name='')
    dsh = m.addVars([(t, p) for t in trucks for p in periods], vtype=GRB.BINARY, name='')
    w = m.addVars([(t, p) for t in trucks for p in periods], vtype=GRB.CONTINUOUS, lb=0, name='')
    obj = gurobipy.quicksum((S_t[t] * s[t, p] + C_t[t] * w[t, p] for t in trucks for p in periods))
    m.setObjective(obj, GRB.MINIMIZE)
    for t in trucks:
        m.addConstr(a[t, 1] == s[t, 1], name=f'act_dyn_start_{t}')
        for p in periods[1:]:
            m.addConstr(a[t, p] == a[t, p - 1] + s[t, p] - dsh[t, p - 1], name=f'act_dyn_{t}_{p}')
    for t in trucks:
        for p in periods[1:]:
            m.addConstr(s[t, p] <= 1 - a[t, p - 1], name=f'start_limit_{t}_{p}')
        m.addConstr(s[t, 4] == 0, name=f'no_start_p4_{t}')
    for t in trucks:
        for p in periods[:3]:
            m.addConstr(dsh[t, p] <= a[t, p], name=f'dsh_limit_{t}_{p}')
        for p in periods[:3]:
            m.addConstr(a[t, p + 1] <= 1 - dsh[t, p], name=f'dsh_next_{t}_{p}')
        for p in periods[:2]:
            m.addConstr(a[t, p + 2] <= 1 - dsh[t, p], name=f'dsh_next2_{t}_{p}')
    for t in trucks:
        for p in periods[:2]:
            m.addConstr(a[t, p] <= a[t, p + 1], name=f'min_consec_{t}_{p}')
    for t in trucks:
        for p in periods:
            m.addConstr(w[t, p] <= Q_t[t] * a[t, p], name=f'cap_limit_{t}_{p}')
    for t in trucks:
        m.addConstr(w[t, 1] <= 300, name=f'load_change_start_{t}')
        for p in periods[1:]:
            m.addConstr(w[t, p] - w[t, p - 1] <= 300, name=f'load_change_up_{t}_{p}')
            m.addConstr(w[t, p - 1] - w[t, p] <= 300, name=f'load_change_down_{t}_{p}')
    for p in periods:
        m.addConstr(gurobipy.quicksum((w[t, p] for t in trucks)) >= d_p[p], name=f'demand_{p}')
    for p in periods:
        m.addConstr(gurobipy.quicksum((w[t, p] for t in trucks)) <= 0.9 * gurobipy.quicksum((Q_t[t] * a[t, p] for t in trucks)), name=f'spare_cap_{p}')
    m.optimize()
    return m
m = solve_problem()