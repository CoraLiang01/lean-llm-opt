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
             'role': 'overall development capacity',
             'table_id': 'file_0_view_0'},
            {'columns': ['ProductName', 'Value', 'Weight'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {'conditions': [{'column': 'ProductName',
                                         'dtype': 'string',
                                         'evidence': '"such as Queens and Brooklyn"',
                                         'operator': 'in',
                                         'value': ['Queens', 'Brooklyn']}],
                         'logic': 'or'},
             'original_rows': 20,
             'records': [{'source_row': 0, 'values': {'ProductName': 'Queens', 'Value': '469', 'Weight': '954'}},
                         {'source_row': 1, 'values': {'ProductName': 'Brooklyn', 'Value': '290', 'Weight': '650'}}],
             'returned_rows': 2,
             'role': 'real estate areas and benefits',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    CSVQA_DATA = {'ignored_file_indices': [], 'query': 'A property developer is planning to develop real estate in New York.There are several areas to choose from, such as Queens and Brooklyn.However, due to limited resources and focus, the developer is not able to develop all the real estate in all the areas and has to select a few for development.The development benefit data for real estate in each area is recorded in the ‘products.csv’ file.The developer has an overall development capacity limit, which is detailed in the ‘capacity.csv’ file.The goal is to decide how many real estate in each areas are to be developed, in order to maximise the overall benefits while adhering to the overall development capacity.The decision variable x_i represents the scale of development per day in area i.', 'relationships': [], 'route': 'RA', 'tables': [{'columns': ['Capacity'], 'file_index': 0, 'file_name': 'capacity.csv', 'filters': {'conditions': [], 'logic': 'and'}, 'original_rows': 1, 'records': [{'source_row': 0, 'values': {'Capacity': '586'}}], 'returned_rows': 1, 'role': 'overall development capacity', 'table_id': 'file_0_view_0'}, {'columns': ['ProductName', 'Value', 'Weight'], 'file_index': 1, 'file_name': 'products.csv', 'filters': {'conditions': [{'column': 'ProductName', 'dtype': 'string', 'evidence': '"such as Queens and Brooklyn"', 'operator': 'in', 'value': ['Queens', 'Brooklyn']}], 'logic': 'or'}, 'original_rows': 20, 'records': [{'source_row': 0, 'values': {'ProductName': 'Queens', 'Value': '469', 'Weight': '954'}}, {'source_row': 1, 'values': {'ProductName': 'Brooklyn', 'Value': '290', 'Weight': '650'}}], 'returned_rows': 2, 'role': 'real estate areas and benefits', 'table_id': 'file_1_view_0'}], 'validation': {'matrix_checks': [], 'status': 'OK'}}
    products_table = None
    for t in CSVQA_DATA['tables']:
        if t['table_id'] == 'file_1_view_0':
            products_table = t
            break
    if products_table is None:
        raise RuntimeError('Missing products table (file_1_view_0)')
    I = []
    b = {}
    w = {}
    for rec in products_table['records']:
        name = rec['values']['ProductName']
        try:
            value = float(rec['values']['Value'])
            weight = float(rec['values']['Weight'])
        except Exception:
            raise RuntimeError(f'Non-numeric Value or Weight for area {name}')
        I.append(name)
        b[name] = value
        w[name] = weight
    capacity_table = None
    for t in CSVQA_DATA['tables']:
        if t['table_id'] == 'file_0_view_0':
            capacity_table = t
            break
    if capacity_table is None:
        raise RuntimeError('Missing capacity table (file_0_view_0)')
    if not capacity_table['records']:
        raise RuntimeError('No records in capacity table')
    try:
        C = float(capacity_table['records'][0]['values']['Capacity'])
    except Exception:
        raise RuntimeError('Non-numeric Capacity value')
    m = gp.Model('NY_RealEstate_Dev')
    m.Params.MIPGap = 0.0001
    x = m.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((b[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((w[i] * x[i] for i in I)) <= C, name='capacity')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()