CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A public health agency wants to choose a fixed number of clinic sites to serve neighborhood demand. '
          'Neighborhood demand is given in neighborhood_demand.csv, travel distances from candidate clinics to '
          'neighborhoods are given in clinic_distance.csv, and the number of clinics that must be opened is given in '
          'planning_parameters.csv.\n'
          '\n'
          'Formulate a p-median location model. For each candidate clinic i, define y_i as a binary variable equal to '
          '1 if clinic i is opened. For each clinic i and neighborhood j, define x_ij as a binary variable equal to 1 '
          'if neighborhood j is assigned to clinic i. The objective is to minimize total demand-weighted travel '
          'distance. The model should assign every neighborhood to exactly one clinic, open exactly the required '
          'number of clinics, link assignments to opened clinics, and impose binary restrictions on all variables.',
 'relationships': [],
 'route': 'FLP',
 'tables': [{'columns': ['Neighborhood', 'Demand'],
             'file_index': 0,
             'file_name': 'neighborhood_demand.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0, 'values': {'Demand': '30', 'Neighborhood': 'N1'}},
                         {'source_row': 1, 'values': {'Demand': '45', 'Neighborhood': 'N2'}},
                         {'source_row': 2, 'values': {'Demand': '25', 'Neighborhood': 'N3'}},
                         {'source_row': 3, 'values': {'Demand': '50', 'Neighborhood': 'N4'}},
                         {'source_row': 4, 'values': {'Demand': '40', 'Neighborhood': 'N5'}},
                         {'source_row': 5, 'values': {'Demand': '35', 'Neighborhood': 'N6'}},
                         {'source_row': 6, 'values': {'Demand': '55', 'Neighborhood': 'N7'}},
                         {'source_row': 7, 'values': {'Demand': '20', 'Neighborhood': 'N8'}},
                         {'source_row': 8, 'values': {'Demand': '60', 'Neighborhood': 'N9'}},
                         {'source_row': 9, 'values': {'Demand': '30', 'Neighborhood': 'N10'}}],
             'returned_rows': 10,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['Clinic', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9', 'N10'],
             'file_index': 1,
             'file_name': 'clinic_distance.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 6,
             'records': [{'source_row': 0,
                          'values': {'Clinic': 'K1',
                                     'N1': '2',
                                     'N10': '16',
                                     'N2': '3',
                                     'N3': '9',
                                     'N4': '10',
                                     'N5': '11',
                                     'N6': '12',
                                     'N7': '13',
                                     'N8': '14',
                                     'N9': '15'}},
                         {'source_row': 1,
                          'values': {'Clinic': 'K2',
                                     'N1': '3',
                                     'N10': '15',
                                     'N2': '2',
                                     'N3': '8',
                                     'N4': '9',
                                     'N5': '10',
                                     'N6': '11',
                                     'N7': '12',
                                     'N8': '13',
                                     'N9': '14'}},
                         {'source_row': 2,
                          'values': {'Clinic': 'K3',
                                     'N1': '10',
                                     'N10': '13',
                                     'N2': '9',
                                     'N3': '2',
                                     'N4': '3',
                                     'N5': '4',
                                     'N6': '9',
                                     'N7': '10',
                                     'N8': '11',
                                     'N9': '12'}},
                         {'source_row': 3,
                          'values': {'Clinic': 'K4',
                                     'N1': '11',
                                     'N10': '12',
                                     'N2': '10',
                                     'N3': '3',
                                     'N4': '2',
                                     'N5': '5',
                                     'N6': '8',
                                     'N7': '9',
                                     'N8': '10',
                                     'N9': '11'}},
                         {'source_row': 4,
                          'values': {'Clinic': 'K5',
                                     'N1': '13',
                                     'N10': '9',
                                     'N2': '12',
                                     'N3': '10',
                                     'N4': '9',
                                     'N5': '8',
                                     'N6': '2',
                                     'N7': '3',
                                     'N8': '4',
                                     'N9': '8'}},
                         {'source_row': 5,
                          'values': {'Clinic': 'K6',
                                     'N1': '14',
                                     'N10': '2',
                                     'N2': '13',
                                     'N3': '11',
                                     'N4': '10',
                                     'N5': '9',
                                     'N6': '3',
                                     'N7': '2',
                                     'N8': '5',
                                     'N9': '3'}}],
             'returned_rows': 6,
             'role': 'file_1',
             'table_id': 'file_1_view_0'},
            {'columns': ['Parameter', 'Value'],
             'file_index': 2,
             'file_name': 'planning_parameters.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 1,
             'records': [{'source_row': 0, 'values': {'Parameter': 'NumberOfClinicsToOpen', 'Value': '3'}}],
             'returned_rows': 1,
             'role': 'file_2',
             'table_id': 'file_2_view_0'}],
 'validation': {'fallback_reason': "Filter lacks query evidence: {'column': 'Parameter', 'operator': 'eq', 'dtype': "
                                   "'string', 'value': 'NumberOfClinicsToOpen', 'evidence': 'NumberOfClinicsToOpen', "
                                   "'format': None, 'inclusive': 'both'}",
                'planner_errors': ["Filter lacks query evidence: {'column': 'Parameter', 'operator': 'eq', 'dtype': "
                                   "'string', 'value': 'NumberOfClinicsToOpen', 'evidence': 'NumberOfClinicsToOpen', "
                                   "'format': None, 'inclusive': 'both'}"],
                'status': 'FALLBACK_FULL_DATA'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    nd_table = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    demand = {}
    for rec in nd_table['records']:
        neigh = rec['values']['Neighborhood']
        dem_val = float(rec['values']['Demand'])
        demand[neigh] = dem_val
    neighborhood_ids = list(demand.keys())
    cd_table = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
    distances = {}
    for rec in cd_table['records']:
        clinic = rec['values']['Clinic']
        distances[clinic] = {}
        for (col, val) in rec['values'].items():
            if col == 'Clinic':
                continue
            distances[clinic][col] = float(val)
    clinic_ids = list(distances.keys())
    pp_table = next((t for t in tables if t['table_id'] == 'file_2_view_0'))
    num_clinics_to_open = None
    for rec in pp_table['records']:
        if rec['values']['Parameter'] == 'NumberOfClinicsToOpen':
            num_clinics_to_open = int(rec['values']['Value'])
            break
    if num_clinics_to_open is None:
        raise ValueError('NumberOfClinicsToOpen parameter not found')
    for clinic in clinic_ids:
        if clinic not in distances:
            raise ValueError(f'Missing distance data for clinic {clinic}')
        for neigh in neighborhood_ids:
            if neigh not in distances[clinic]:
                raise ValueError(f'Missing distance from clinic {clinic} to neighborhood {neigh}')
    for neigh in neighborhood_ids:
        if neigh not in demand:
            raise ValueError(f'Missing demand for neighborhood {neigh}')
    if num_clinics_to_open > len(clinic_ids):
        raise ValueError('Number of clinics to open exceeds available clinics')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    y = model.addVars(clinic_ids, vtype=GRB.BINARY, name='')
    x = model.addVars([(i, j) for i in clinic_ids for j in neighborhood_ids], vtype=GRB.BINARY, name='')
    obj = gurobipy.quicksum((demand[j] * distances[i][j] * x[i, j] for i in clinic_ids for j in neighborhood_ids))
    model.setObjective(obj, GRB.MINIMIZE)
    for j in neighborhood_ids:
        model.addConstr(gurobipy.quicksum((x[i, j] for i in clinic_ids)) == 1, name=f'assign_{j}')
    for i in clinic_ids:
        model.addConstr(gurobipy.quicksum((x[i, j] for j in neighborhood_ids)) <= y[i], name=f'link_{i}')
    model.addConstr(gurobipy.quicksum((y[i] for i in clinic_ids)) == num_clinics_to_open, name='open_count')
    model.optimize()
    return model
m = solve_problem()