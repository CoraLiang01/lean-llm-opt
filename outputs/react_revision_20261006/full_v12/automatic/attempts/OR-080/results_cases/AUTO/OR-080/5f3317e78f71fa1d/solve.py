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
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
import pandas as pd
import numpy as np

def solve_problem(CSVQA_FRAMES):
    frame = CSVQA_FRAMES['file_0_view_0']
    trucks = []
    Q = {}
    S = {}
    C = {}
    for (idx, row) in frame.iterrows():
        truck_id = row['truck_id']
        trucks.append(truck_id)
        Q[truck_id] = float(row['Q'])
        S[truck_id] = float(row['S'])
        C[truck_id] = float(row['C'])
    periods = [1, 2, 3, 4]
    d = {}
    d[1] = float(frame.iloc[0]['d1'])
    d[2] = float(frame.iloc[0]['d2'])
    d[3] = float(frame.iloc[0]['d3'])
    d[4] = float(frame.iloc[0]['d4'])
    M = max(Q.values())
    m = gp.Model('TruckScheduling')
    x_keys = [(i, t) for i in trucks for t in periods]
    y_keys = [(i, t) for i in trucks for t in periods]
    z_keys = [(i, t) for i in trucks for t in periods]
    x_vars = m.addVars(x_keys, lb=0, vtype=gp.GRB.CONTINUOUS, name='')
    y_vars = m.addVars(y_keys, vtype=gp.GRB.BINARY, name='')
    z_vars = m.addVars(z_keys, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((S[i] * z_vars[i, t] + C[i] * x_vars[i, t] for i in trucks for t in periods)), gp.GRB.MINIMIZE)
    for t in periods:
        m.addConstr(gp.quicksum((x_vars[i, t] for i in trucks)) >= d[t], name=f'demand_{t}')
    for t in periods:
        m.addConstr(gp.quicksum((x_vars[i, t] for i in trucks)) <= 0.9 * gp.quicksum((Q[i] * y_vars[i, t] for i in trucks)), name=f'spare_capacity_{t}')
    for i in trucks:
        for t in periods:
            m.addConstr(x_vars[i, t] <= Q[i] * y_vars[i, t], name=f'truck_capacity_{i}_{t}')
            m.addConstr(x_vars[i, t] >= 0, name=f'x_nonneg_{i}_{t}')
    for i in trucks:
        m.addConstr(z_vars[i, 1] >= y_vars[i, 1], name=f'startup1_{i}')
    for i in trucks:
        for t in [2, 3, 4]:
            m.addConstr(z_vars[i, t] >= y_vars[i, t] - y_vars[i, t - 1] if t > 1 else z_vars[i, t] >= y_vars[i, t], name=f'startup_logic_{i}_{t}')
    for i in trucks:
        m.addConstr(z_vars[i, 4] == 0, name=f'no_startup_4_{i}')
    for i in trucks:
        for t in [1, 2, 3]:
            m.addConstr(y_vars[i, t + 1] >= y_vars[i, t] - z_vars[i, t], name=f'min_up_{i}_{t}')
    for i in trucks:
        m.addConstr(y_vars[i, 1] - y_vars[i, 2] <= 1 - y_vars[i, 3], name=f'min_down_2_{i}')
        m.addConstr(y_vars[i, 2] - y_vars[i, 3] <= 1 - y_vars[i, 4], name=f'min_down_3_{i}')
    for i in trucks:
        for t in [2, 3, 4]:
            m.addConstr(x_vars[i, t] - x_vars[i, t - 1] <= 300, name=f'ramp_up_{i}_{t}')
            m.addConstr(x_vars[i, t - 1] - x_vars[i, t] <= 300, name=f'ramp_down_{i}_{t}')
    for i in trucks:
        x_vars[i, 0] = 0.0
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)