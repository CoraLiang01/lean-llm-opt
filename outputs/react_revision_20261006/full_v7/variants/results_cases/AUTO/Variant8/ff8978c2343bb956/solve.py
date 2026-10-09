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
 'relationships': [{'column_axis': {'id_column': 'Neighborhood', 'table_id': 'file_0_view_0'},
                    'matrix_table_id': 'file_1_view_0',
                    'row_axis': {'id_column': 'Clinic', 'table_id': 'file_1_view_0'},
                    'row_id_column': 'Clinic',
                    'type': 'matrix'}],
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
             'role': 'neighborhood demand',
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
             'role': 'clinic-to-neighborhood distance matrix',
             'table_id': 'file_1_view_0'},
            {'columns': ['Parameter', 'Value'],
             'file_index': 2,
             'file_name': 'planning_parameters.csv',
             'filters': {'conditions': [{'column': 'Parameter',
                                         'dtype': 'string',
                                         'evidence': 'the number of clinics that must be opened is given in '
                                                     'planning_parameters.csv',
                                         'operator': 'exact',
                                         'value': 'NumberOfClinicsToOpen'}],
                         'logic': 'and'},
             'original_rows': 1,
             'records': [{'source_row': 0, 'values': {'Parameter': 'NumberOfClinicsToOpen', 'Value': '3'}}],
             'returned_rows': 1,
             'role': 'planning parameters',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [{'column_ids_aligned': True,
                                   'column_mapping_basis': 'exact',
                                   'expected_shape': [6, 10],
                                   'matrix_table_id': 'file_1_view_0',
                                   'row_ids_aligned': True,
                                   'row_mapping_basis': 'exact',
                                   'shape': [6, 10]}],
                'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem(CSVQA_FRAMES):
    df_neigh = CSVQA_FRAMES['file_0_view_0']
    df_dist = CSVQA_FRAMES['file_1_view_0']
    df_param = CSVQA_FRAMES['file_2_view_0']
    clinics = list(df_dist['Clinic'])
    neighborhoods = list(df_neigh['Neighborhood'])
    demand = {}
    for (_, row) in df_neigh.iterrows():
        j = row['Neighborhood']
        try:
            demand[j] = float(row['Demand'])
        except Exception:
            raise ValueError(f"Non-numeric demand for neighborhood {j}: {row['Demand']}")
    distance = {}
    for (_, row) in df_dist.iterrows():
        i = row['Clinic']
        distance[i] = {}
        for j in neighborhoods:
            val = row[j]
            try:
                distance[i][j] = float(val)
            except Exception:
                raise ValueError(f'Non-numeric distance for clinic {i}, neighborhood {j}: {val}')
    param_row = df_param[df_param['Parameter'].str.casefold() == 'numberofclinicstoopen']
    if param_row.empty:
        raise ValueError("Parameter 'NumberOfClinicsToOpen' not found in planning_parameters.csv")
    try:
        p = int(float(param_row.iloc[0]['Value']))
    except Exception:
        raise ValueError(f"Non-numeric value for NumberOfClinicsToOpen: {param_row.iloc[0]['Value']}")
    for i in clinics:
        if i not in distance or set(distance[i].keys()) != set(neighborhoods):
            raise ValueError(f'Distance data missing for clinic {i} or neighborhoods mismatch')
    if set(demand.keys()) != set(neighborhoods):
        raise ValueError('Demand data missing for some neighborhoods')
    m = gp.Model('p_median')
    m.Params.MIPGap = 0.0001
    y_vars = m.addVars(clinics, vtype=GRB.BINARY, name='')
    x_vars = m.addVars([(i, j) for i in clinics for j in neighborhoods], vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((demand[j] * distance[i][j] * x_vars[i, j] for i in clinics for j in neighborhoods)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x_vars[i, j] for i in clinics)) == 1 for j in neighborhoods), name='')
    m.addConstr(gp.quicksum((y_vars[i] for i in clinics)) == p, name='open_p')
    m.addConstrs((x_vars[i, j] <= y_vars[i] for i in clinics for j in neighborhoods), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem(CSVQA_FRAMES)