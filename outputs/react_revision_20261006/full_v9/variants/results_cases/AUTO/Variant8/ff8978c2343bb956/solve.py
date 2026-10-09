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
 'validation': {'fallback_reason': "Matrix validation failed: {'matrix_table_id': 'file_1_view_0', 'shape': [6, 10], "
                                   "'expected_shape': [6, 6], 'row_ids_aligned': True, 'column_ids_aligned': False, "
                                   "'row_mapping_basis': 'exact', 'column_mapping_basis': 'unresolved'}",
                'planner_errors': ["Matrix validation failed: {'matrix_table_id': 'file_1_view_0', 'shape': [6, 10], "
                                   "'expected_shape': [6, 6], 'row_ids_aligned': True, 'column_ids_aligned': False, "
                                   "'row_mapping_basis': 'exact', 'column_mapping_basis': 'unresolved'}"],
                'status': 'FALLBACK_FULL_DATA'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem(CSVQA_FRAMES):
    neighborhoods_frame = CSVQA_FRAMES['file_0_view_0']
    neighborhoods = []
    demand = {}
    for (_, row) in neighborhoods_frame.iterrows():
        n = row['Neighborhood']
        neighborhoods.append(n)
        try:
            demand[n] = float(row['Demand'])
        except Exception:
            raise ValueError(f"Invalid demand value for neighborhood {n}: {row['Demand']}")
    clinics_frame = CSVQA_FRAMES['file_1_view_0']
    clinics = []
    for (_, row) in clinics_frame.iterrows():
        c = row['Clinic']
        clinics.append(c)
    distance = {}
    for (_, row) in clinics_frame.iterrows():
        i = row['Clinic']
        distance[i] = {}
        for j in neighborhoods:
            try:
                distance[i][j] = float(row[j])
            except Exception:
                raise ValueError(f'Missing or invalid distance for clinic {i}, neighborhood {j}: {row[j]}')
    params_frame = CSVQA_FRAMES['file_2_view_0']
    p = None
    for (_, row) in params_frame.iterrows():
        if str(row['Parameter']).casefold() == 'numberofclinicstoopen':
            try:
                p = int(float(row['Value']))
            except Exception:
                raise ValueError(f"Invalid value for NumberOfClinicsToOpen: {row['Value']}")
    if p is None:
        raise ValueError('NumberOfClinicsToOpen parameter not found.')
    for i in clinics:
        if i not in distance:
            raise ValueError(f'Clinic {i} missing in distance data.')
        for j in neighborhoods:
            if j not in distance[i]:
                raise ValueError(f'Distance missing for clinic {i}, neighborhood {j}.')
    m = gp.Model('p_median_location')
    m.Params.MIPGap = 0.0001
    y_vars = m.addVars(clinics, vtype=GRB.BINARY, name='')
    x_vars = m.addVars(clinics, neighborhoods, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((demand[j] * distance[i][j] * x_vars[i, j] for i in clinics for j in neighborhoods)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x_vars[i, j] for i in clinics)) == 1 for j in neighborhoods), name='')
    m.addConstr(gp.quicksum((y_vars[i] for i in clinics)) == p, name='open_p')
    m.addConstrs((x_vars[i, j] <= y_vars[i] for i in clinics for j in neighborhoods), name='')
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)