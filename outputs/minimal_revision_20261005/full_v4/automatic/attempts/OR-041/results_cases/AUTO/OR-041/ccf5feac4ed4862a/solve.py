CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A property developer is planning to develop real estate in New York.There are several areas to choose from, '
          'such as Queens and Brooklyn.However, due to limited resources and focus, the developer is not able to '
          'develop all the real estate in all the areas and has to select a few for development.The development '
          'benefit data for real estate in each area is recorded in the ‘products.csv’ file.The developer has an '
          'overall development capacity limit, which is detailed in the ‘capacity.csv’ file.The goal is to decide how '
          'many real estate in each areas are to be developed, in order to maximise the overall benefits while '
          'adhering to the overall development capacity.The decision variable x_i represents the scale of development '
          'per day in area i.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'columns': ['Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 1,
             'records': [{'source_row': 0, 'values': {'Capacity': '586'}}],
             'returned_rows': 1,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['ProductName', 'Value', 'Weight'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 20,
             'records': [{'source_row': 0, 'values': {'ProductName': 'Queens', 'Value': '469', 'Weight': '954'}},
                         {'source_row': 1, 'values': {'ProductName': 'Brooklyn', 'Value': '290', 'Weight': '650'}},
                         {'source_row': 2, 'values': {'ProductName': 'Manhattan', 'Value': '236', 'Weight': '961'}},
                         {'source_row': 3, 'values': {'ProductName': 'Bronx', 'Value': '235', 'Weight': '950'}},
                         {'source_row': 4, 'values': {'ProductName': 'Staten Island', 'Value': '745', 'Weight': '379'}},
                         {'source_row': 5, 'values': {'ProductName': 'Harlem', 'Value': '684', 'Weight': '776'}},
                         {'source_row': 6,
                          'values': {'ProductName': 'Upper East Side', 'Value': '444', 'Weight': '381'}},
                         {'source_row': 7,
                          'values': {'ProductName': 'Lower Manhattan', 'Value': '172', 'Weight': '808'}},
                         {'source_row': 8, 'values': {'ProductName': 'Midtown', 'Value': '1000', 'Weight': '937'}},
                         {'source_row': 9,
                          'values': {'ProductName': 'Long Island City', 'Value': '336', 'Weight': '608'}},
                         {'source_row': 10, 'values': {'ProductName': 'Williamsburg', 'Value': '546', 'Weight': '912'}},
                         {'source_row': 11, 'values': {'ProductName': 'Bushwick', 'Value': '535', 'Weight': '391'}},
                         {'source_row': 12, 'values': {'ProductName': 'Flatbush', 'Value': '539', 'Weight': '465'}},
                         {'source_row': 13, 'values': {'ProductName': 'Greenpoint', 'Value': '831', 'Weight': '490'}},
                         {'source_row': 14, 'values': {'ProductName': 'Park Slope', 'Value': '139', 'Weight': '918'}},
                         {'source_row': 15, 'values': {'ProductName': 'Astoria', 'Value': '432', 'Weight': '787'}},
                         {'source_row': 16,
                          'values': {'ProductName': 'Jackson Heights', 'Value': '627', 'Weight': '347'}},
                         {'source_row': 17, 'values': {'ProductName': 'Flushing', 'Value': '629', 'Weight': '274'}},
                         {'source_row': 18, 'values': {'ProductName': 'Sunnyside', 'Value': '292', 'Weight': '642'}},
                         {'source_row': 19, 'values': {'ProductName': 'Ditmars', 'Value': '978', 'Weight': '130'}}],
             'returned_rows': 20,
             'role': 'file_1',
             'table_id': 'file_1_view_0'}],
 'validation': {'fallback_reason': 'Illustrative query examples cannot justify a source-row filter: areas to choose '
                                   'from, such as Queens and Brooklyn',
                'planner_errors': ['Illustrative query examples cannot justify a source-row filter: areas to choose '
                                   'from, such as Queens and Brooklyn'],
                'status': 'FALLBACK_FULL_DATA'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    products_records = [{'ProductName': 'Queens', 'Value': '469', 'Weight': '954'}, {'ProductName': 'Brooklyn', 'Value': '290', 'Weight': '650'}, {'ProductName': 'Manhattan', 'Value': '236', 'Weight': '961'}, {'ProductName': 'Bronx', 'Value': '235', 'Weight': '950'}, {'ProductName': 'Staten Island', 'Value': '745', 'Weight': '379'}, {'ProductName': 'Harlem', 'Value': '684', 'Weight': '776'}, {'ProductName': 'Upper East Side', 'Value': '444', 'Weight': '381'}, {'ProductName': 'Lower Manhattan', 'Value': '172', 'Weight': '808'}, {'ProductName': 'Midtown', 'Value': '1000', 'Weight': '937'}, {'ProductName': 'Long Island City', 'Value': '336', 'Weight': '608'}, {'ProductName': 'Williamsburg', 'Value': '546', 'Weight': '912'}, {'ProductName': 'Bushwick', 'Value': '535', 'Weight': '391'}, {'ProductName': 'Flatbush', 'Value': '539', 'Weight': '465'}, {'ProductName': 'Greenpoint', 'Value': '831', 'Weight': '490'}, {'ProductName': 'Park Slope', 'Value': '139', 'Weight': '918'}, {'ProductName': 'Astoria', 'Value': '432', 'Weight': '787'}, {'ProductName': 'Jackson Heights', 'Value': '627', 'Weight': '347'}, {'ProductName': 'Flushing', 'Value': '629', 'Weight': '274'}, {'ProductName': 'Sunnyside', 'Value': '292', 'Weight': '642'}, {'ProductName': 'Ditmars', 'Value': '978', 'Weight': '130'}]
    capacity_record = {'Capacity': '586'}
    I = [rec['ProductName'] for rec in products_records]
    v = {rec['ProductName']: int(rec['Value']) for rec in products_records}
    w = {rec['ProductName']: int(rec['Weight']) for rec in products_records}
    C = int(capacity_record['Capacity'])
    m = gp.Model('NY_RealEstate_Development')
    x = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((w[i] * x[i] for i in I)) <= C, name='capacity')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()