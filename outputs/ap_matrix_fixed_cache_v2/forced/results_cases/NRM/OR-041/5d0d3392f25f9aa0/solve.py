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
                                         'evidence': 'Queens and Brooklyn',
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
CSVQA_DATA = {'ignored_file_indices': [], 'query': 'A property developer is planning to develop real estate in New York.There are several areas to choose from, such as Queens and Brooklyn.However, due to limited resources and focus, the developer is not able to develop all the real estate in all the areas and has to select a few for development.The development benefit data for real estate in each area is recorded in the ‘products.csv’ file.The developer has an overall development capacity limit, which is detailed in the ‘capacity.csv’ file.The goal is to decide how many real estate in each areas are to be developed, in order to maximise the overall benefits while adhering to the overall development capacity.The decision variable x_i represents the scale of development per day in area i.', 'relationships': [], 'route': 'NRM', 'tables': [{'columns': ['Capacity'], 'file_index': 0, 'file_name': 'capacity.csv', 'filters': {'conditions': [], 'logic': 'and'}, 'original_rows': 1, 'records': [{'source_row': 0, 'values': {'Capacity': '586'}}], 'returned_rows': 1, 'role': 'overall development capacity', 'table_id': 'file_0_view_0'}, {'columns': ['ProductName', 'Value', 'Weight'], 'file_index': 1, 'file_name': 'products.csv', 'filters': {'conditions': [{'column': 'ProductName', 'dtype': 'string', 'evidence': 'Queens and Brooklyn', 'operator': 'in', 'value': ['Queens', 'Brooklyn']}], 'logic': 'or'}, 'original_rows': 20, 'records': [{'source_row': 0, 'values': {'ProductName': 'Queens', 'Value': '469', 'Weight': '954'}}, {'source_row': 1, 'values': {'ProductName': 'Brooklyn', 'Value': '290', 'Weight': '650'}}], 'returned_rows': 2, 'role': 'development benefit per area', 'table_id': 'file_1_view_0'}], 'validation': {'matrix_checks': [], 'status': 'OK'}}
products_table = None
capacity_table = None
for t in CSVQA_DATA['tables']:
    if t['table_id'] == 'file_1_view_0':
        products_table = t
    if t['table_id'] == 'file_0_view_0':
        capacity_table = t
if products_table is None or capacity_table is None:
    raise ValueError('Required data tables not found in CSVQA_DATA.')
areas = []
benefit = {}
weight = {}
for rec in products_table['records']:
    name = rec['values']['ProductName']
    areas.append(name)
    try:
        benefit[name] = float(rec['values']['Value'])
        weight[name] = float(rec['values']['Weight'])
    except Exception as e:
        raise ValueError(f'Invalid numeric value in products.csv for area {name}: {e}')
if len(capacity_table['records']) != 1:
    raise ValueError('Expected exactly one record in capacity.csv.')
try:
    capacity = float(capacity_table['records'][0]['values']['Capacity'])
except Exception as e:
    raise ValueError(f'Invalid numeric value in capacity.csv: {e}')
for name in areas:
    if name not in benefit or name not in weight:
        raise ValueError(f'Missing benefit or weight for area {name}.')
m = gp.Model('Original_RAG_NRM')
x_vars = m.addVars(areas, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((benefit[i] * x_vars[i] for i in areas)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[i] * x_vars[i] for i in areas)) <= capacity, name='capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')