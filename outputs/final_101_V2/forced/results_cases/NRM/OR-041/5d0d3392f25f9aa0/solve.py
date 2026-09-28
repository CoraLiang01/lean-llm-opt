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
 'route': 'NRM',
 'tables': [{'columns': ['Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {},
             'original_rows': 1,
             'records': [{'source_row': 0, 'values': {'Capacity': '586'}}],
             'returned_rows': 1,
             'role': 'overall development capacity',
             'table_id': 'file_0_view_0'},
            {'columns': ['ProductName', 'Value', 'Weight'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {},
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
             'role': 'development benefit per area',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB
CSVQA_DATA = {'ignored_file_indices': [], 'query': 'A property developer is planning to develop real estate in New York.There are several areas to choose from, such as Queens and Brooklyn.However, due to limited resources and focus, the developer is not able to develop all the real estate in all the areas and has to select a few for development.The development benefit data for real estate in each area is recorded in the ‘products.csv’ file.The developer has an overall development capacity limit, which is detailed in the ‘capacity.csv’ file.The goal is to decide how many real estate in each areas are to be developed, in order to maximise the overall benefits while adhering to the overall development capacity.The decision variable x_i represents the scale of development per day in area i.', 'relationships': [], 'route': 'NRM', 'tables': [{'columns': ['Capacity'], 'file_index': 0, 'file_name': 'capacity.csv', 'filters': {}, 'original_rows': 1, 'records': [{'source_row': 0, 'values': {'Capacity': '586'}}], 'returned_rows': 1, 'role': 'overall development capacity', 'table_id': 'file_0_view_0'}, {'columns': ['ProductName', 'Value', 'Weight'], 'file_index': 1, 'file_name': 'products.csv', 'filters': {}, 'original_rows': 20, 'records': [{'source_row': 0, 'values': {'ProductName': 'Queens', 'Value': '469', 'Weight': '954'}}, {'source_row': 1, 'values': {'ProductName': 'Brooklyn', 'Value': '290', 'Weight': '650'}}, {'source_row': 2, 'values': {'ProductName': 'Manhattan', 'Value': '236', 'Weight': '961'}}, {'source_row': 3, 'values': {'ProductName': 'Bronx', 'Value': '235', 'Weight': '950'}}, {'source_row': 4, 'values': {'ProductName': 'Staten Island', 'Value': '745', 'Weight': '379'}}, {'source_row': 5, 'values': {'ProductName': 'Harlem', 'Value': '684', 'Weight': '776'}}, {'source_row': 6, 'values': {'ProductName': 'Upper East Side', 'Value': '444', 'Weight': '381'}}, {'source_row': 7, 'values': {'ProductName': 'Lower Manhattan', 'Value': '172', 'Weight': '808'}}, {'source_row': 8, 'values': {'ProductName': 'Midtown', 'Value': '1000', 'Weight': '937'}}, {'source_row': 9, 'values': {'ProductName': 'Long Island City', 'Value': '336', 'Weight': '608'}}, {'source_row': 10, 'values': {'ProductName': 'Williamsburg', 'Value': '546', 'Weight': '912'}}, {'source_row': 11, 'values': {'ProductName': 'Bushwick', 'Value': '535', 'Weight': '391'}}, {'source_row': 12, 'values': {'ProductName': 'Flatbush', 'Value': '539', 'Weight': '465'}}, {'source_row': 13, 'values': {'ProductName': 'Greenpoint', 'Value': '831', 'Weight': '490'}}, {'source_row': 14, 'values': {'ProductName': 'Park Slope', 'Value': '139', 'Weight': '918'}}, {'source_row': 15, 'values': {'ProductName': 'Astoria', 'Value': '432', 'Weight': '787'}}, {'source_row': 16, 'values': {'ProductName': 'Jackson Heights', 'Value': '627', 'Weight': '347'}}, {'source_row': 17, 'values': {'ProductName': 'Flushing', 'Value': '629', 'Weight': '274'}}, {'source_row': 18, 'values': {'ProductName': 'Sunnyside', 'Value': '292', 'Weight': '642'}}, {'source_row': 19, 'values': {'ProductName': 'Ditmars', 'Value': '978', 'Weight': '130'}}], 'returned_rows': 20, 'role': 'development benefit per area', 'table_id': 'file_1_view_0'}], 'validation': {'matrix_checks': [], 'status': 'OK'}}
products_table = None
for t in CSVQA_DATA['tables']:
    if t['table_id'] == 'file_1_view_0':
        products_table = t
        break
if products_table is None:
    raise ValueError('products.csv (file_1_view_0) not found in CSVQA_DATA.')
areas = []
v = {}
w = {}
for rec in products_table['records']:
    name = rec['values']['ProductName']
    try:
        value = float(rec['values']['Value'])
        weight = float(rec['values']['Weight'])
    except Exception:
        raise ValueError(f'Non-numeric Value or Weight for area {name}')
    areas.append(name)
    v[name] = value
    w[name] = weight
capacity_table = None
for t in CSVQA_DATA['tables']:
    if t['table_id'] == 'file_0_view_0':
        capacity_table = t
        break
if capacity_table is None:
    raise ValueError('capacity.csv (file_0_view_0) not found in CSVQA_DATA.')
if len(capacity_table['records']) != 1:
    raise ValueError('capacity.csv must have exactly one row for overall capacity.')
try:
    C = float(capacity_table['records'][0]['values']['Capacity'])
except Exception:
    raise ValueError('Non-numeric Capacity in capacity.csv.')
if set(v.keys()) != set(areas) or set(w.keys()) != set(areas):
    raise ValueError('Mismatch in area indices between Value and Weight.')
m = gp.Model('Original_RAG_NRM')
x = m.addVars(areas, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((v[i] * x[i] for i in areas)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((w[i] * x[i] for i in areas)) <= C, name='capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for variable in m.getVars():
        print(f'{variable.VarName}: {variable.X}')
else:
    print(f'Solver status: {m.Status}')