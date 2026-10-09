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
 'tables': [{'columns': ['previous_period_capacity', 'Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 1,
             'records': [{'source_row': 0, 'values': {'Capacity': '586', 'previous_period_capacity': '681'}}],
             'returned_rows': 1,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['previous_period_development_status',
                         'previous_period_unit_value',
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
                                     'previous_period_development_status': 'Planned',
                                     'previous_period_unit_value': '458'}},
                         {'source_row': 1,
                          'values': {'ProductName': 'Brooklyn',
                                     'Value': '290',
                                     'Weight': '650',
                                     'previous_period_development_status': 'Planned',
                                     'previous_period_unit_value': '275'}},
                         {'source_row': 2,
                          'values': {'ProductName': 'Manhattan',
                                     'Value': '236',
                                     'Weight': '961',
                                     'previous_period_development_status': 'Planned',
                                     'previous_period_unit_value': '202'}},
                         {'source_row': 3,
                          'values': {'ProductName': 'Bronx',
                                     'Value': '235',
                                     'Weight': '950',
                                     'previous_period_development_status': 'Completed',
                                     'previous_period_unit_value': '257'}},
                         {'source_row': 4,
                          'values': {'ProductName': 'Staten Island',
                                     'Value': '745',
                                     'Weight': '379',
                                     'previous_period_development_status': 'Planned',
                                     'previous_period_unit_value': '741'}},
                         {'source_row': 5,
                          'values': {'ProductName': 'Harlem',
                                     'Value': '684',
                                     'Weight': '776',
                                     'previous_period_development_status': 'Completed',
                                     'previous_period_unit_value': '657'}},
                         {'source_row': 6,
                          'values': {'ProductName': 'Upper East Side',
                                     'Value': '444',
                                     'Weight': '381',
                                     'previous_period_development_status': 'In progress',
                                     'previous_period_unit_value': '374'}},
                         {'source_row': 7,
                          'values': {'ProductName': 'Lower Manhattan',
                                     'Value': '172',
                                     'Weight': '808',
                                     'previous_period_development_status': 'In progress',
                                     'previous_period_unit_value': '201'}},
                         {'source_row': 8,
                          'values': {'ProductName': 'Midtown',
                                     'Value': '1000',
                                     'Weight': '937',
                                     'previous_period_development_status': 'In progress',
                                     'previous_period_unit_value': '901'}},
                         {'source_row': 9,
                          'values': {'ProductName': 'Long Island City',
                                     'Value': '336',
                                     'Weight': '608',
                                     'previous_period_development_status': 'Planned',
                                     'previous_period_unit_value': '277'}},
                         {'source_row': 10,
                          'values': {'ProductName': 'Williamsburg',
                                     'Value': '546',
                                     'Weight': '912',
                                     'previous_period_development_status': 'Planned',
                                     'previous_period_unit_value': '547'}},
                         {'source_row': 11,
                          'values': {'ProductName': 'Bushwick',
                                     'Value': '535',
                                     'Weight': '391',
                                     'previous_period_development_status': 'In progress',
                                     'previous_period_unit_value': '549'}},
                         {'source_row': 12,
                          'values': {'ProductName': 'Flatbush',
                                     'Value': '539',
                                     'Weight': '465',
                                     'previous_period_development_status': 'Completed',
                                     'previous_period_unit_value': '432'}},
                         {'source_row': 13,
                          'values': {'ProductName': 'Greenpoint',
                                     'Value': '831',
                                     'Weight': '490',
                                     'previous_period_development_status': 'Completed',
                                     'previous_period_unit_value': '985'}},
                         {'source_row': 14,
                          'values': {'ProductName': 'Park Slope',
                                     'Value': '139',
                                     'Weight': '918',
                                     'previous_period_development_status': 'In progress',
                                     'previous_period_unit_value': '137'}},
                         {'source_row': 15,
                          'values': {'ProductName': 'Astoria',
                                     'Value': '432',
                                     'Weight': '787',
                                     'previous_period_development_status': 'Completed',
                                     'previous_period_unit_value': '449'}},
                         {'source_row': 16,
                          'values': {'ProductName': 'Jackson Heights',
                                     'Value': '627',
                                     'Weight': '347',
                                     'previous_period_development_status': 'Completed',
                                     'previous_period_unit_value': '570'}},
                         {'source_row': 17,
                          'values': {'ProductName': 'Flushing',
                                     'Value': '629',
                                     'Weight': '274',
                                     'previous_period_development_status': 'Planned',
                                     'previous_period_unit_value': '582'}},
                         {'source_row': 18,
                          'values': {'ProductName': 'Sunnyside',
                                     'Value': '292',
                                     'Weight': '642',
                                     'previous_period_development_status': 'Completed',
                                     'previous_period_unit_value': '259'}},
                         {'source_row': 19,
                          'values': {'ProductName': 'Ditmars',
                                     'Value': '978',
                                     'Weight': '130',
                                     'previous_period_development_status': 'In progress',
                                     'previous_period_unit_value': '1025'}}],
             'returned_rows': 20,
             'role': 'file_1',
             'table_id': 'file_1_view_0'}],
 'validation': {'fallback_reason': 'Illustrative query examples cannot justify a source-row filter: areas to choose '
                                   'from, such as Queens and Brooklyn',
                'planner_errors': ['Illustrative query examples cannot justify a source-row filter: areas to choose '
                                   'from, such as Queens and Brooklyn'],
                'status': 'FALLBACK_FULL_DATA'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem(CSVQA_FRAMES):
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
    if 'Capacity' not in capacity_frame.columns or capacity_frame.shape[0] == 0:
        raise ValueError('Capacity not found in capacity_frame')
    try:
        C = float(capacity_frame.iloc[0]['Capacity'])
    except Exception:
        raise ValueError(f"Non-numeric Capacity: {capacity_frame.iloc[0]['Capacity']}")
    if set(v.keys()) != set(I) or set(w.keys()) != set(I):
        raise ValueError('Mismatch in index set and parameter keys')
    m = gp.Model('NY_RealEstate_Development')
    m.Params.MIPGap = 0.0001
    x_vars = m.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((v[i] * x_vars[i] for i in I)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((w[i] * x_vars[i] for i in I)) <= C, name='capacity')
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for variable in m.getVars():
        print(f'{variable.VarName}: {variable.X}')
else:
    print(f'Solver status: {m.Status}')