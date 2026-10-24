CSVQA_DATA = {'query': "A city wants to choose emergency service centers so that all demand districts are covered. Candidate center data are provided in service_centers.csv, including each center's opening cost and the districts that can be covered from that center. The full set of districts that must be covered is listed in districts.csv.\n\nFormulate a binary set-covering model. For each candidate center i, define y_i as a binary variable equal to 1 if center i is opened and 0 otherwise. The objective is to minimize total opening cost. The model should include coverage constraints requiring every district to be covered by at least one opened center and binary restrictions on all opening variables.", 'relationships': [], 'tables': [{'columns': ['Center', 'OpeningCost', 'CoveredDistricts'], 'file_name': 'service_centers.csv', 'records': [{'data_row_number': 0, 'values': {'Center': 'SC1', 'CoveredDistricts': 'D1;D2;D4', 'OpeningCost': '12'}}, {'data_row_number': 1, 'values': {'Center': 'SC2', 'CoveredDistricts': 'D2;D3;D5', 'OpeningCost': '15'}}, {'data_row_number': 2, 'values': {'Center': 'SC3', 'CoveredDistricts': 'D4;D5;D6', 'OpeningCost': '18'}}, {'data_row_number': 3, 'values': {'Center': 'SC4', 'CoveredDistricts': 'D6;D7', 'OpeningCost': '10'}}, {'data_row_number': 4, 'values': {'Center': 'SC5', 'CoveredDistricts': 'D7;D8;D10', 'OpeningCost': '14'}}, {'data_row_number': 5, 'values': {'Center': 'SC6', 'CoveredDistricts': 'D8;D9', 'OpeningCost': '13'}}, {'data_row_number': 6, 'values': {'Center': 'SC7', 'CoveredDistricts': 'D1;D9;D10', 'OpeningCost': '16'}}, {'data_row_number': 7, 'values': {'Center': 'SC8', 'CoveredDistricts': 'D3;D4;D8', 'OpeningCost': '11'}}], 'table_id': 'file_0_view_0'}, {'columns': ['District'], 'file_name': 'districts.csv', 'records': [{'data_row_number': 0, 'values': {'District': 'D1'}}, {'data_row_number': 1, 'values': {'District': 'D2'}}, {'data_row_number': 2, 'values': {'District': 'D3'}}, {'data_row_number': 3, 'values': {'District': 'D4'}}, {'data_row_number': 4, 'values': {'District': 'D5'}}, {'data_row_number': 5, 'values': {'District': 'D6'}}, {'data_row_number': 6, 'values': {'District': 'D7'}}, {'data_row_number': 7, 'values': {'District': 'D8'}}, {'data_row_number': 8, 'values': {'District': 'D9'}}, {'data_row_number': 9, 'values': {'District': 'D10'}}], 'table_id': 'file_1_view_0'}]}
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
