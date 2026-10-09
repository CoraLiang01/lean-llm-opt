CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A property developer is planning to develop real estate in New York.There are several areas to choose from, '
          'such as Queens and Brooklyn.However, due to limited resources and focus, the developer is not able to '
          'develop all the real estate in all the areas and has to select a few for development.The development '
          'benefit data for real estate in each area is recorded in the ‚Äòproducts.csv‚Äô file.The developer has an '
          'overall development capacity limit, which is detailed in the ‚Äòcapacity.csv‚Äô file.The goal is to decide '
          'how many real estate in each areas are to be developed, in order to maximise the overall benefits while '
          'adhering to the overall development capacity.The decision variable x_i represents the scale of development '
          'per day in area i.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'columns': ['archive_revision_number', 'Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 1,
             'records': [{'source_row': 0, 'values': {'Capacity': '586', 'archive_revision_number': '7'}}],
             'returned_rows': 1,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['record_keeper_group',
                         'archived_attachment_count',
                         'archive_revision_number',
                         'ProductName',
                         'Value',
                         'Weight'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 20,
             'records': [{'source_row': 0,
                          'values': {'ProductName': 'Queens',
                                     'Value': '469',
                                     'Weight': '954',
                                     'archive_revision_number': '9',
                                     'archived_attachment_count': '6',
                                     'record_keeper_group': 'Team B'}},
                         {'source_row': 1,
                          'values': {'ProductName': 'Brooklyn',
                                     'Value': '290',
                                     'Weight': '650',
                                     'archive_revision_number': '6',
                                     'archived_attachment_count': '3',
                                     'record_keeper_group': 'Team C'}},
                         {'source_row': 2,
                          'values': {'ProductName': 'Manhattan',
                                     'Value': '236',
                                     'Weight': '961',
                                     'archive_revision_number': '1',
                                     'archived_attachment_count': '1',
                                     'record_keeper_group': 'Team C'}},
                         {'source_row': 3,
                          'values': {'ProductName': 'Bronx',
                                     'Value': '235',
                                     'Weight': '950',
                                     'archive_revision_number': '2',
                                     'archived_attachment_count': '1',
                                     'record_keeper_group': 'Team B'}},
                         {'source_row': 4,
                          'values': {'ProductName': 'Staten Island',
                                     'Value': '745',
                                     'Weight': '379',
                                     'archive_revision_number': '6',
                                     'archived_attachment_count': '6',
                                     'record_keeper_group': 'Team B'}},
                         {'source_row': 5,
                          'values': {'ProductName': 'Harlem',
                                     'Value': '684',
                                     'Weight': '776',
                                     'archive_revision_number': '9',
                                     'archived_attachment_count': '3',
                                     'record_keeper_group': 'Team B'}},
                         {'source_row': 6,
                          'values': {'ProductName': 'Upper East Side',
                                     'Value': '444',
                                     'Weight': '381',
                                     'archive_revision_number': '2',
                                     'archived_attachment_count': '1',
                                     'record_keeper_group': 'Team A'}},
                         {'source_row': 7,
                          'values': {'ProductName': 'Lower Manhattan',
                                     'Value': '172',
                                     'Weight': '808',
                                     'archive_revision_number': '5',
                                     'archived_attachment_count': '3',
                                     'record_keeper_group': 'Team C'}},
                         {'source_row': 8,
                          'values': {'ProductName': 'Midtown',
                                     'Value': '1000',
                                     'Weight': '937',
                                     'archive_revision_number': '9',
                                     'archived_attachment_count': '2',
                                     'record_keeper_group': 'Team B'}},
                         {'source_row': 9,
                          'values': {'ProductName': 'Long Island City',
                                     'Value': '336',
                                     'Weight': '608',
                                     'archive_revision_number': '1',
                                     'archived_attachment_count': '3',
                                     'record_keeper_group': 'Team B'}},
                         {'source_row': 10,
                          'values': {'ProductName': 'Williamsburg',
                                     'Value': '546',
                                     'Weight': '912',
                                     'archive_revision_number': '3',
                                     'archived_attachment_count': '2',
                                     'record_keeper_group': 'Team B'}},
                         {'source_row': 11,
                          'values': {'ProductName': 'Bushwick',
                                     'Value': '535',
                                     'Weight': '391',
                                     'archive_revision_number': '1',
                                     'archived_attachment_count': '6',
                                     'record_keeper_group': 'Team C'}},
                         {'source_row': 12,
                          'values': {'ProductName': 'Flatbush',
                                     'Value': '539',
                                     'Weight': '465',
                                     'archive_revision_number': '5',
                                     'archived_attachment_count': '3',
                                     'record_keeper_group': 'Team A'}},
                         {'source_row': 13,
                          'values': {'ProductName': 'Greenpoint',
                                     'Value': '831',
                                     'Weight': '490',
                                     'archive_revision_number': '7',
                                     'archived_attachment_count': '2',
                                     'record_keeper_group': 'Team C'}},
                         {'source_row': 14,
                          'values': {'ProductName': 'Park Slope',
                                     'Value': '139',
                                     'Weight': '918',
                                     'archive_revision_number': '6',
                                     'archived_attachment_count': '1',
                                     'record_keeper_group': 'Team A'}},
                         {'source_row': 15,
                          'values': {'ProductName': 'Astoria',
                                     'Value': '432',
                                     'Weight': '787',
                                     'archive_revision_number': '1',
                                     'archived_attachment_count': '6',
                                     'record_keeper_group': 'Team B'}},
                         {'source_row': 16,
                          'values': {'ProductName': 'Jackson Heights',
                                     'Value': '627',
                                     'Weight': '347',
                                     'archive_revision_number': '1',
                                     'archived_attachment_count': '6',
                                     'record_keeper_group': 'Team C'}},
                         {'source_row': 17,
                          'values': {'ProductName': 'Flushing',
                                     'Value': '629',
                                     'Weight': '274',
                                     'archive_revision_number': '7',
                                     'archived_attachment_count': '2',
                                     'record_keeper_group': 'Team C'}},
                         {'source_row': 18,
                          'values': {'ProductName': 'Sunnyside',
                                     'Value': '292',
                                     'Weight': '642',
                                     'archive_revision_number': '6',
                                     'archived_attachment_count': '4',
                                     'record_keeper_group': 'Team A'}},
                         {'source_row': 19,
                          'values': {'ProductName': 'Ditmars',
                                     'Value': '978',
                                     'Weight': '130',
                                     'archive_revision_number': '7',
                                     'archived_attachment_count': '2',
                                     'record_keeper_group': 'Team B'}}],
             'returned_rows': 20,
             'role': 'file_1',
             'table_id': 'file_1_view_0'}],
 'validation': {'fallback_reason': 'Illustrative query examples cannot justify a source-row filter: such as Queens and '
                                   'Brooklyn',
                'planner_errors': ['Illustrative query examples cannot justify a source-row filter: such as Queens and '
                                   'Brooklyn'],
                'status': 'FALLBACK_FULL_DATA'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    products_frame = CSVQA_FRAMES['file_1_view_0']
    capacity_frame = CSVQA_FRAMES['file_0_view_0']
    I = []
    v = {}
    w = {}
    for (source_row, row) in products_frame.iterrows():
        area = row['ProductName']
        I.append(area)
        try:
            v[area] = float(row['Value'])
        except Exception:
            raise ValueError(f"Non-numeric Value for area {area}: {row['Value']}")
        try:
            w[area] = float(row['Weight'])
        except Exception:
            raise ValueError(f"Non-numeric Weight for area {area}: {row['Weight']}")
    if capacity_frame.shape[0] != 1:
        raise ValueError('Expected exactly one row in capacity.csv for total capacity.')
    try:
        C = float(capacity_frame.iloc[0]['Capacity'])
    except Exception:
        raise ValueError(f"Non-numeric Capacity: {capacity_frame.iloc[0]['Capacity']}")
    m = gp.Model('property_development')
    x_vars = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v[i] * x_vars[i] for i in I)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((w[i] * x_vars[i] for i in I)) <= C, name='capacity')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for variable in m.getVars():
            print(f'{variable.VarName}: {variable.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()