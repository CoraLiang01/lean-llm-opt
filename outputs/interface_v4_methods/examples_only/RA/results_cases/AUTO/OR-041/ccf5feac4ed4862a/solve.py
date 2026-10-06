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
                                         'evidence': 'areas to choose from, such as Queens and Brooklyn',
                                         'inclusive': 'both',
                                         'operator': 'in',
                                         'value': ['Queens', 'Brooklyn']}],
                         'logic': 'or'},
             'original_rows': 20,
             'records': [{'source_row': 0, 'values': {'ProductName': 'Queens', 'Value': '469', 'Weight': '954'}},
                         {'source_row': 1, 'values': {'ProductName': 'Brooklyn', 'Value': '290', 'Weight': '650'}}],
             'returned_rows': 2,
             'role': 'development options and benefits',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    data = CSVQA_DATA
    products_table = None
    for t in data['tables']:
        if t['table_id'] == 'file_1_view_0':
            products_table = t
            break
    if products_table is None:
        raise ValueError('products.csv (file_1_view_0) not found in data.')
    capacity_table = None
    for t in data['tables']:
        if t['table_id'] == 'file_0_view_0':
            capacity_table = t
            break
    if capacity_table is None:
        raise ValueError('capacity.csv (file_0_view_0) not found in data.')
    I = []
    b = {}
    w = {}
    for rec in products_table['records']:
        pname = rec['values']['ProductName']
        try:
            b_i = float(rec['values']['Value'])
            w_i = float(rec['values']['Weight'])
        except Exception as e:
            raise ValueError(f'Invalid Value or Weight for area {pname}: {e}')
        I.append(pname)
        b[pname] = b_i
        w[pname] = w_i
    if len(capacity_table['records']) != 1:
        raise ValueError('Expected exactly one row in capacity.csv (file_0_view_0)')
    try:
        C = float(capacity_table['records'][0]['values']['Capacity'])
    except Exception as e:
        raise ValueError(f'Invalid Capacity value: {e}')
    for i in I:
        if i not in b or i not in w:
            raise ValueError(f'Missing data for area {i}')
    m = gp.Model('property_development')
    m.Params.MIPGap = 0.0001
    x = m.addVars(I, vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((b[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((w[i] * x[i] for i in I)) <= C, name='capacity')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for i in I:
            print(f'{x[i].VarName} {x[i].X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()