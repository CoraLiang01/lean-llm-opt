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
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 1,
             'records': [{'source_row': 0, 'values': {'Capacity': '586'}}],
             'returned_rows': 1,
             'role': 'overall development capacity',
             'table_id': 'file_0_view_0'},
            {'columns': ['ProductName', 'Value', 'Weight'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {'conditions': [{'column': 'ProductName',
                                         'dtype': 'string',
                                         'evidence': 'such as Queens and Brooklyn',
                                         'operator': 'in',
                                         'value': ['Queens', 'Brooklyn']}],
                         'logic': 'or'},
             'original_rows': 20,
             'records': [{'source_row': 0, 'values': {'ProductName': 'Queens', 'Value': '469', 'Weight': '954'}},
                         {'source_row': 1, 'values': {'ProductName': 'Brooklyn', 'Value': '290', 'Weight': '650'}}],
             'returned_rows': 2,
             'role': 'development benefit per area',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB
tables = CSVQA_DATA['tables']
capacity_table = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
capacity_records = capacity_table['records']
if len(capacity_records) != 1:
    raise ValueError('Expected exactly one capacity record in file_0_view_0.')
try:
    C = float(capacity_records[0]['values']['Capacity'])
except Exception:
    raise ValueError('Capacity value missing or not convertible to float.')
products_table = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
products_records = products_table['records']
areas = []
b = {}
w = {}
for rec in products_records:
    vals = rec['values']
    area = vals['ProductName']
    try:
        b_i = float(vals['Value'])
        w_i = float(vals['Weight'])
    except Exception:
        raise ValueError(f'Missing or invalid Value/Weight for area {area}.')
    areas.append(area)
    b[area] = b_i
    w[area] = w_i
if not areas:
    raise ValueError('No areas found in products data.')
for area in areas:
    if area not in b or area not in w:
        raise ValueError(f'Missing coefficients for area {area}.')
m = gp.Model('NY_RealEstate_Dev')
x_vars = m.addVars(areas, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((b[i] * x_vars[i] for i in areas)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((w[i] * x_vars[i] for i in areas)) <= C, name='capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')