CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A utility company must place emergency repair depots so that every service zone is covered by at least one '
          'selected depot. Candidate depot opening costs and covered service zones are listed in facility_sites.csv, '
          'and the full set of service zones is listed in service_zones.csv.\n'
          '\n'
          'Formulate a minimum-cost set covering model. For each candidate depot i, define y_i as a binary variable '
          'equal to 1 if depot i is opened. The objective is to minimize total opening cost. The model should include '
          'one coverage constraint for each service zone requiring at least one opened depot that covers that zone, '
          'and binary restrictions for all depot-opening variables.',
 'relationships': [],
 'route': 'Others',
 'tables': [{'columns': ['Center', 'OpeningCost', 'CoveredDistricts'],
             'file_index': 0,
             'file_name': 'facility_sites.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 8,
             'records': [{'source_row': 0,
                          'values': {'Center': 'B1', 'CoveredDistricts': 'Z1;Z2;Z5', 'OpeningCost': '11'}},
                         {'source_row': 1,
                          'values': {'Center': 'B2', 'CoveredDistricts': 'Z2;Z3;Z6', 'OpeningCost': '14'}},
                         {'source_row': 2,
                          'values': {'Center': 'B3', 'CoveredDistricts': 'Z4;Z5;Z8', 'OpeningCost': '10'}},
                         {'source_row': 3,
                          'values': {'Center': 'B4', 'CoveredDistricts': 'Z1;Z6;Z7', 'OpeningCost': '13'}},
                         {'source_row': 4,
                          'values': {'Center': 'B5', 'CoveredDistricts': 'Z3;Z7;Z9', 'OpeningCost': '16'}},
                         {'source_row': 5,
                          'values': {'Center': 'B6', 'CoveredDistricts': 'Z8;Z9;Z10', 'OpeningCost': '9'}},
                         {'source_row': 6,
                          'values': {'Center': 'B7', 'CoveredDistricts': 'Z4;Z10', 'OpeningCost': '12'}},
                         {'source_row': 7,
                          'values': {'Center': 'B8', 'CoveredDistricts': 'Z5;Z6;Z9', 'OpeningCost': '15'}}],
             'returned_rows': 8,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['Zone'],
             'file_index': 1,
             'file_name': 'service_zones.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0, 'values': {'Zone': 'Z1'}},
                         {'source_row': 1, 'values': {'Zone': 'Z2'}},
                         {'source_row': 2, 'values': {'Zone': 'Z3'}},
                         {'source_row': 3, 'values': {'Zone': 'Z4'}},
                         {'source_row': 4, 'values': {'Zone': 'Z5'}},
                         {'source_row': 5, 'values': {'Zone': 'Z6'}},
                         {'source_row': 6, 'values': {'Zone': 'Z7'}},
                         {'source_row': 7, 'values': {'Zone': 'Z8'}},
                         {'source_row': 8, 'values': {'Zone': 'Z9'}},
                         {'source_row': 9, 'values': {'Zone': 'Z10'}}],
             'returned_rows': 10,
             'role': 'file_1',
             'table_id': 'file_1_view_0'}],
 'validation': {'fallback_reason': "Relationship references an unknown table_id: {'type': 'matrix', 'matrix_table_id': "
                                   "'file_2_view_0', 'row_id_column': 'Zone', 'row_axis': {'table_id': "
                                   "'file_1_view_0', 'id_column': 'Zone'}, 'column_axis': {'table_id': "
                                   "'file_0_view_0', 'id_column': 'Center'}}",
                'planner_errors': ["Relationship references an unknown table_id: {'type': 'matrix', 'matrix_table_id': "
                                   "'file_2_view_0', 'row_id_column': 'Zone', 'row_axis': {'table_id': "
                                   "'file_1_view_0', 'id_column': 'Zone'}, 'column_axis': {'table_id': "
                                   "'file_0_view_0', 'id_column': 'Center'}}"],
                'status': 'FALLBACK_FULL_DATA'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    facility_table = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    service_table = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
    I = []
    c = {}
    coverage = {}
    for rec in facility_table['records']:
        center = rec['values']['Center']
        if center in I:
            raise ValueError(f'Duplicate depot identifier {center}')
        I.append(center)
        opening_cost = float(rec['values']['OpeningCost'])
        c[center] = opening_cost
        covered_str = rec['values']['CoveredDistricts']
        zones = [z.strip() for z in covered_str.split(';') if z.strip()]
        coverage[center] = set(zones)
    J = []
    for rec in service_table['records']:
        zone = rec['values']['Zone']
        if zone in J:
            raise ValueError(f'Duplicate zone identifier {zone}')
        J.append(zone)
    for zone in J:
        if not any((zone in coverage[center] for center in I)):
            raise ValueError(f'Zone {zone} is not covered by any depot')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    y = m.addVars(I, vtype=GRB.BINARY, name='')
    m.setObjective(gurobipy.quicksum((c[i] * y[i] for i in I)), GRB.MINIMIZE)
    for zone in J:
        m.addConstr(gurobipy.quicksum((y[i] for i in I if zone in coverage[i])) >= 1, name=f'cover_{zone}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'Optimal objective value: {m.ObjVal:.6f}')
        for var in m.getVars():
            print(f'{var.VarName} = {var.X:.6f}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()