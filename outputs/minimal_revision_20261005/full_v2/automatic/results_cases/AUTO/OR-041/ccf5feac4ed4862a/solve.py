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
                                         'evidence': 'such as Queens and Brooklyn',
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
    areas = []
    v = {}
    w = {}
    for rec in CSVQA_DATA['tables']:
        if rec['table_id'] == 'file_1_view_0':
            for row in rec['records']:
                pname = row['values']['ProductName']
                areas.append(pname)
                v[pname] = int(row['values']['Value'])
                w[pname] = int(row['values']['Weight'])
    C = None
    for rec in CSVQA_DATA['tables']:
        if rec['table_id'] == 'file_0_view_0':
            for row in rec['records']:
                C = int(row['values']['Capacity'])
    if C is None:
        raise ValueError('Missing overall development capacity C')
    if set(areas) != set(v.keys()) or set(areas) != set(w.keys()):
        raise ValueError('Mismatch in area identifiers and coefficients')
    m = gp.Model('NY_RealEstate_Development')
    x = m.addVars(areas, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v[i] * x[i] for i in areas)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((w[i] * x[i] for i in areas)) <= C, name='capacity')
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