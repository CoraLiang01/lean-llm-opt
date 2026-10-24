CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A water authority must install monitoring sensors so every monitoring zone is covered. Candidate sensor '
          'costs and covered zones are listed in sensor_sites.csv, and the complete set of monitoring zones is listed '
          'in monitoring_zones.csv.\n'
          '\n'
          'Formulate a minimum-cost set covering model. For each candidate sensor site i, define y_i as a binary '
          'variable equal to 1 if a sensor is installed at site i. The objective is to minimize total installation '
          'cost. The model should include one coverage constraint for each monitoring zone requiring at least one '
          'selected site that covers that zone, and binary restrictions for all sensor-site variables.',
 'relationships': [],
 'route': 'Others',
 'tables': [{'columns': ['Center', 'OpeningCost', 'CoveredDistricts'],
             'file_index': 0,
             'file_name': 'sensor_sites.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 9,
             'records': [{'source_row': 0,
                          'values': {'Center': 'S1', 'CoveredDistricts': 'Q1;Q2;Q4', 'OpeningCost': '8'}},
                         {'source_row': 1,
                          'values': {'Center': 'S2', 'CoveredDistricts': 'Q2;Q3;Q5', 'OpeningCost': '12'}},
                         {'source_row': 2,
                          'values': {'Center': 'S3', 'CoveredDistricts': 'Q4;Q6;Q7', 'OpeningCost': '11'}},
                         {'source_row': 3,
                          'values': {'Center': 'S4', 'CoveredDistricts': 'Q5;Q8', 'OpeningCost': '10'}},
                         {'source_row': 4,
                          'values': {'Center': 'S5', 'CoveredDistricts': 'Q6;Q8;Q9', 'OpeningCost': '13'}},
                         {'source_row': 5,
                          'values': {'Center': 'S6', 'CoveredDistricts': 'Q7;Q10;Q11', 'OpeningCost': '9'}},
                         {'source_row': 6,
                          'values': {'Center': 'S7', 'CoveredDistricts': 'Q1;Q9;Q10', 'OpeningCost': '14'}},
                         {'source_row': 7,
                          'values': {'Center': 'S8', 'CoveredDistricts': 'Q3;Q5;Q11', 'OpeningCost': '7'}},
                         {'source_row': 8,
                          'values': {'Center': 'S9', 'CoveredDistricts': 'Q2;Q6;Q10', 'OpeningCost': '15'}}],
             'returned_rows': 9,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'},
            {'columns': ['Zone'],
             'file_index': 1,
             'file_name': 'monitoring_zones.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 11,
             'records': [{'source_row': 0, 'values': {'Zone': 'Q1'}},
                         {'source_row': 1, 'values': {'Zone': 'Q2'}},
                         {'source_row': 2, 'values': {'Zone': 'Q3'}},
                         {'source_row': 3, 'values': {'Zone': 'Q4'}},
                         {'source_row': 4, 'values': {'Zone': 'Q5'}},
                         {'source_row': 5, 'values': {'Zone': 'Q6'}},
                         {'source_row': 6, 'values': {'Zone': 'Q7'}},
                         {'source_row': 7, 'values': {'Zone': 'Q8'}},
                         {'source_row': 8, 'values': {'Zone': 'Q9'}},
                         {'source_row': 9, 'values': {'Zone': 'Q10'}},
                         {'source_row': 10, 'values': {'Zone': 'Q11'}}],
             'returned_rows': 11,
             'role': 'descriptive non-unique role',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    sensor_table = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    zone_table = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
    I = []
    c = {}
    coverage = {}
    for rec in sensor_table['records']:
        values = rec['values']
        site = values['Center']
        cost = float(values['OpeningCost'])
        I.append(site)
        c[site] = cost
        covered = values['CoveredDistricts'].split(';') if values['CoveredDistricts'] else []
        coverage[site] = set(covered)
    J = []
    for rec in zone_table['records']:
        zone = rec['values']['Zone']
        J.append(zone)
    zone_to_sites = {j: [] for j in J}
    for site in I:
        for zone in coverage[site]:
            if zone in zone_to_sites:
                zone_to_sites[zone].append(site)
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    y = m.addVars(I, vtype=GRB.BINARY, name='')
    m.setObjective(gurobipy.quicksum((c[i] * y[i] for i in I)), GRB.MINIMIZE)
    for (j, sites) in zone_to_sites.items():
        m.addConstr(gurobipy.quicksum((y[i] for i in sites)) >= 1, name=f'cover_{j}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(m.ObjVal)
        for var in m.getVars():
            print(var.VarName, var.X)
    else:
        print(m.Status)
    return m
m = solve_problem()