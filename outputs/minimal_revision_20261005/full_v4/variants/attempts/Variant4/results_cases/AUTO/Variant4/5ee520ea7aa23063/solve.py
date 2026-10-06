CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A city wants to choose emergency service centers so that all demand districts are covered. Candidate center '
          "data are provided in service_centers.csv, including each center's opening cost and the districts that can "
          'be covered from that center. The full set of districts that must be covered is listed in districts.csv.\n'
          '\n'
          'Formulate a binary set-covering model. For each candidate center i, define y_i as a binary variable equal '
          'to 1 if center i is opened and 0 otherwise. The objective is to minimize total opening cost. The model '
          'should include coverage constraints requiring every district to be covered by at least one opened center '
          'and binary restrictions on all opening variables.',
 'relationships': [],
 'route': 'FLP',
 'tables': [{'columns': ['Center', 'OpeningCost', 'CoveredDistricts'],
             'file_index': 0,
             'file_name': 'service_centers.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 8,
             'records': [{'source_row': 0,
                          'values': {'Center': 'SC1', 'CoveredDistricts': 'D1;D2;D4', 'OpeningCost': '12'}},
                         {'source_row': 1,
                          'values': {'Center': 'SC2', 'CoveredDistricts': 'D2;D3;D5', 'OpeningCost': '15'}},
                         {'source_row': 2,
                          'values': {'Center': 'SC3', 'CoveredDistricts': 'D4;D5;D6', 'OpeningCost': '18'}},
                         {'source_row': 3,
                          'values': {'Center': 'SC4', 'CoveredDistricts': 'D6;D7', 'OpeningCost': '10'}},
                         {'source_row': 4,
                          'values': {'Center': 'SC5', 'CoveredDistricts': 'D7;D8;D10', 'OpeningCost': '14'}},
                         {'source_row': 5,
                          'values': {'Center': 'SC6', 'CoveredDistricts': 'D8;D9', 'OpeningCost': '13'}},
                         {'source_row': 6,
                          'values': {'Center': 'SC7', 'CoveredDistricts': 'D1;D9;D10', 'OpeningCost': '16'}},
                         {'source_row': 7,
                          'values': {'Center': 'SC8', 'CoveredDistricts': 'D3;D4;D8', 'OpeningCost': '11'}}],
             'returned_rows': 8,
             'role': 'candidate service centers',
             'table_id': 'file_0_view_0'},
            {'columns': ['District'],
             'file_index': 1,
             'file_name': 'districts.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0, 'values': {'District': 'D1'}},
                         {'source_row': 1, 'values': {'District': 'D2'}},
                         {'source_row': 2, 'values': {'District': 'D3'}},
                         {'source_row': 3, 'values': {'District': 'D4'}},
                         {'source_row': 4, 'values': {'District': 'D5'}},
                         {'source_row': 5, 'values': {'District': 'D6'}},
                         {'source_row': 6, 'values': {'District': 'D7'}},
                         {'source_row': 7, 'values': {'District': 'D8'}},
                         {'source_row': 8, 'values': {'District': 'D9'}},
                         {'source_row': 9, 'values': {'District': 'D10'}}],
             'returned_rows': 10,
             'role': 'demand districts',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    data = {'service_centers': CSVQA_DATA['tables'][0]['records'], 'districts': CSVQA_DATA['tables'][1]['records']}
    I = [rec['values']['Center'] for rec in data['service_centers']]
    J = [rec['values']['District'] for rec in data['districts']]
    c = {}
    for rec in data['service_centers']:
        i = rec['values']['Center']
        c[i] = float(rec['values']['OpeningCost'])
    covered = {}
    for rec in data['service_centers']:
        i = rec['values']['Center']
        covered_districts = [d.strip() for d in rec['values']['CoveredDistricts'].split(';') if d.strip()]
        covered[i] = set(covered_districts)
    C = {}
    for i in I:
        for j in J:
            C[i, j] = 1 if j in covered[i] else 0
    for i in I:
        if i not in c:
            raise ValueError(f'Missing opening cost for center {i}')
        if i not in covered:
            raise ValueError(f'Missing covered districts for center {i}')
    for j in J:
        found = any((C[i, j] == 1 for i in I))
        if not found:
            raise ValueError(f'No center covers district {j}')
    m = gp.Model('SetCovering')
    y = m.addVars(I, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((c[i] * y[i] for i in I)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((C[i, j] * y[i] for i in I)) >= 1 for j in J), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')