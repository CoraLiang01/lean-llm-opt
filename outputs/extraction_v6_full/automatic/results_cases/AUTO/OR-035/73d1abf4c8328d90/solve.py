CSVQA_DATA = {'bindings': [{'index_columns': [], 'parameter': 'capacity', 'table_id': 'file_0_view_0', 'value_column': 'Capacity'},
              {'index_columns': ['ProductName'],
               'parameter': 'profit',
               'table_id': 'file_1_view_0',
               'value_column': 'Value'},
              {'index_columns': ['ProductName'],
               'parameter': 'weight',
               'table_id': 'file_1_view_0',
               'value_column': 'Weight'}],
 'ignored_file_indices': [],
 'query': 'A small bakery in South Korea, and each day need to stock up on various types of bread. For each type of '
          'bread, we have an expected profit, which can be found in "products.csv." However, the shop has limited '
          'storage capacity, with details provided in "capacity.csv.".Therefore, we must decide which types of bread '
          'to order each day to maximize our total expected profit while staying within our storage limits. The '
          'decision variables x_i represents the number of units of bread type i to be ordered each day.The decision '
          'variables must be integers.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'columns': ['Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 1,
             'records': [{'source_row': 0, 'values': {'Capacity': '180'}}],
             'returned_rows': 1,
             'role': 'capacity',
             'table_id': 'file_0_view_0'},
            {'columns': ['ProductName', 'Value', 'Weight'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0, 'values': {'ProductName': 'Baguette', 'Value': '888', 'Weight': '4'}},
                         {'source_row': 1, 'values': {'ProductName': 'Croissant', 'Value': '134', 'Weight': '2'}},
                         {'source_row': 2, 'values': {'ProductName': 'Sourdough', 'Value': '129', 'Weight': '4'}},
                         {'source_row': 3, 'values': {'ProductName': 'Rye Bread', 'Value': '370', 'Weight': '3'}},
                         {'source_row': 4, 'values': {'ProductName': 'Brioche', 'Value': '921', 'Weight': '2'}},
                         {'source_row': 5, 'values': {'ProductName': 'Focaccia', 'Value': '765', 'Weight': '1'}},
                         {'source_row': 6, 'values': {'ProductName': 'Ciabatta', 'Value': '154', 'Weight': '2'}},
                         {'source_row': 7, 'values': {'ProductName': 'Pita', 'Value': '837', 'Weight': '1'}},
                         {'source_row': 8, 'values': {'ProductName': 'Bagel', 'Value': '584', 'Weight': '3'}},
                         {'source_row': 9, 'values': {'ProductName': 'English Muffin', 'Value': '365', 'Weight': '3'}}],
             'returned_rows': 10,
             'role': 'products',
             'table_id': 'file_1_view_0'}],
 'validation': {'binding_checks': [{'index_columns': [],
                                    'key_count': 1,
                                    'parameter': 'capacity',
                                    'status': 'OK',
                                    'table_id': 'file_0_view_0',
                                    'value_column': 'Capacity'},
                                   {'index_columns': ['ProductName'],
                                    'key_count': 10,
                                    'parameter': 'profit',
                                    'status': 'OK',
                                    'table_id': 'file_1_view_0',
                                    'value_column': 'Value'},
                                   {'index_columns': ['ProductName'],
                                    'key_count': 10,
                                    'parameter': 'weight',
                                    'status': 'OK',
                                    'table_id': 'file_1_view_0',
                                    'value_column': 'Weight'}],
                'matrix_checks': [],
                'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    capacity_record = CSVQA_DATA['tables'][0]['records'][0]['values']
    C = int(capacity_record['Capacity'])
    product_records = CSVQA_DATA['tables'][1]['records']
    I = []
    p = {}
    w = {}
    for rec in product_records:
        name = rec['values']['ProductName']
        I.append(name)
        p[name] = int(rec['values']['Value'])
        w[name] = int(rec['values']['Weight'])
    m = gp.Model('Bakery_Bread_Stocking')
    x = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='x')
    m.setObjective(gp.quicksum((p[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((w[i] * x[i] for i in I)) <= C, name='storage_capacity')
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