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
             'filters': {},
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
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'},
            {'columns': ['District'],
             'file_index': 1,
             'file_name': 'districts.csv',
             'filters': {},
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
             'role': 'descriptive non-unique role',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    centers_table = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    districts_table = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
    centers = []
    f_i = {}
    coverage = {}
    for rec in centers_table['records']:
        values = rec['values']
        center = values['Center']
        centers.append(center)
        if 'OpeningCost' not in values:
            raise ValueError(f'Missing OpeningCost for center {center}')
        try:
            cost = float(values['OpeningCost'])
        except ValueError:
            raise ValueError(f'Invalid OpeningCost for center {center}')
        f_i[center] = cost
        covered_str = values.get('CoveredDistricts', '')
        if covered_str:
            districts_covered = [d.strip() for d in covered_str.split(';') if d.strip()]
        else:
            districts_covered = []
        coverage[center] = set(districts_covered)
    districts = []
    for rec in districts_table['records']:
        values = rec['values']
        district = values['District']
        districts.append(district)
    for center in centers:
        if center not in f_i:
            raise ValueError(f'Missing opening cost for center {center}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    y = model.addVars(centers, vtype=GRB.BINARY, name='')
    model.setObjective(gurobipy.quicksum((f_i[i] * y[i] for i in centers)), GRB.MINIMIZE)
    model.addConstrs((gurobipy.quicksum((y[i] for i in centers if j in coverage[i])) >= 1 for j in districts), name='')
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(f'ObjVal: {model.ObjVal}')
        for var in model.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {model.Status}')
    return model
m = solve_problem()