CSVQA_DATA = {'bindings': [{'index_columns': ['item_name'],
               'parameter': 'profit',
               'table_id': 'file_1_view_0',
               'value_column': 'item_value'},
              {'index_columns': ['item_name'],
               'parameter': 'resource_requirement',
               'table_id': 'file_1_view_0',
               'value_column': 'resource_requirement'},
              {'index_columns': [],
               'parameter': 'resource_capacity',
               'table_id': 'file_0_view_0',
               'value_column': 'resource_capacity'}],
 'ignored_file_indices': [],
 'query': 'A small bakery in South Korea, and each day need to stock up on various types of bread. For each type of '
          'bread, we have an expected profit, which can be found in "products.csv." However, the shop has limited '
          'storage capacity, with details provided in "capacity.csv.".Therefore, we must decide which types of bread '
          'to order each day to maximize our total expected profit while staying within our storage limits. The '
          'decision variables x_i represents the number of units of bread type i to be ordered each day.The decision '
          'variables must be integers.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'columns': ['resource_capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 1,
             'records': [{'source_row': 0, 'values': {'resource_capacity': '180'}}],
             'returned_rows': 1,
             'role': 'capacity',
             'table_id': 'file_0_view_0'},
            {'columns': ['item_name', 'item_value', 'resource_requirement'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0,
                          'values': {'item_name': 'Baguette', 'item_value': '888', 'resource_requirement': '4'}},
                         {'source_row': 1,
                          'values': {'item_name': 'Croissant', 'item_value': '134', 'resource_requirement': '2'}},
                         {'source_row': 2,
                          'values': {'item_name': 'Sourdough', 'item_value': '129', 'resource_requirement': '4'}},
                         {'source_row': 3,
                          'values': {'item_name': 'Rye Bread', 'item_value': '370', 'resource_requirement': '3'}},
                         {'source_row': 4,
                          'values': {'item_name': 'Brioche', 'item_value': '921', 'resource_requirement': '2'}},
                         {'source_row': 5,
                          'values': {'item_name': 'Focaccia', 'item_value': '765', 'resource_requirement': '1'}},
                         {'source_row': 6,
                          'values': {'item_name': 'Ciabatta', 'item_value': '154', 'resource_requirement': '2'}},
                         {'source_row': 7,
                          'values': {'item_name': 'Pita', 'item_value': '837', 'resource_requirement': '1'}},
                         {'source_row': 8,
                          'values': {'item_name': 'Bagel', 'item_value': '584', 'resource_requirement': '3'}},
                         {'source_row': 9,
                          'values': {'item_name': 'English Muffin', 'item_value': '365', 'resource_requirement': '3'}}],
             'returned_rows': 10,
             'role': 'products',
             'table_id': 'file_1_view_0'}],
 'validation': {'binding_checks': [{'index_columns': ['item_name'],
                                    'key_count': 10,
                                    'parameter': 'profit',
                                    'status': 'OK',
                                    'table_id': 'file_1_view_0',
                                    'value_column': 'item_value'},
                                   {'index_columns': ['item_name'],
                                    'key_count': 10,
                                    'parameter': 'resource_requirement',
                                    'status': 'OK',
                                    'table_id': 'file_1_view_0',
                                    'value_column': 'resource_requirement'},
                                   {'index_columns': [],
                                    'key_count': 1,
                                    'parameter': 'resource_capacity',
                                    'status': 'OK',
                                    'table_id': 'file_0_view_0',
                                    'value_column': 'resource_capacity'}],
                'matrix_checks': [],
                'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    products_table = CSVQA_DATA['tables'][1]
    product_records = products_table['records']
    items = []
    profit = {}
    resource_requirement = {}
    for rec in product_records:
        vals = rec['values']
        item = vals['item_name']
        items.append(item)
        profit[item] = int(vals['item_value'])
        resource_requirement[item] = int(vals['resource_requirement'])
    capacity_table = CSVQA_DATA['tables'][0]
    capacity_record = capacity_table['records'][0]
    C = int(capacity_record['values']['resource_capacity'])
    m = gp.Model('Bakery_Bread_Stocking')
    x = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='x')
    m.setObjective(gp.quicksum((profit[i] * x[i] for i in items)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((resource_requirement[i] * x[i] for i in items)) <= C, name='storage_capacity')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()