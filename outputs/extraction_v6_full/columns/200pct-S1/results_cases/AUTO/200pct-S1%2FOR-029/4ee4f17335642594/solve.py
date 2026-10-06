CSVQA_DATA = {'bindings': [{'index_columns': ['ShelfID'],
               'parameter': 'capacity',
               'table_id': 'file_0_view_0',
               'value_column': 'Capacity'},
              {'index_columns': ['ProductName'],
               'parameter': 'value',
               'table_id': 'file_1_view_0',
               'value_column': 'Value'},
              {'index_columns': ['ProductName'],
               'parameter': 'weight',
               'table_id': 'file_1_view_0',
               'value_column': 'Weight'}],
 'ignored_file_indices': [],
 'query': 'In retail, shops need to allocate various types of products to different displays. The capacity limit of '
          'each display is provided in "capacity.csv", and the value and weight of each product are provided in '
          '"products.csv". The objective is to determine the optimal number of each product to place on each display '
          'so as to maximize the total value of all products placed across the displays, while ensuring that the total '
          'weight of the products on each display does not exceed its capacity. In addition, the total quantity of the '
          'first product placed across all displays must be at least 5. The decision variable x_{ij} represents the '
          'number of units of product j placed on display i.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'columns': ['ShelfID', 'Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0, 'values': {'Capacity': '5', 'ShelfID': '1'}},
                         {'source_row': 1, 'values': {'Capacity': '7', 'ShelfID': '2'}},
                         {'source_row': 2, 'values': {'Capacity': '6', 'ShelfID': '3'}},
                         {'source_row': 3, 'values': {'Capacity': '8', 'ShelfID': '4'}},
                         {'source_row': 4, 'values': {'Capacity': '5.5', 'ShelfID': '5'}},
                         {'source_row': 5, 'values': {'Capacity': '9', 'ShelfID': '6'}},
                         {'source_row': 6, 'values': {'Capacity': '6.5', 'ShelfID': '7'}},
                         {'source_row': 7, 'values': {'Capacity': '7.5', 'ShelfID': '8'}},
                         {'source_row': 8, 'values': {'Capacity': '8.2', 'ShelfID': '9'}},
                         {'source_row': 9, 'values': {'Capacity': '5.7', 'ShelfID': '10'}}],
             'returned_rows': 10,
             'role': 'display_capacity',
             'table_id': 'file_0_view_0'},
            {'columns': ['ProductName', 'Value', 'Weight'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 20,
             'records': [{'source_row': 0, 'values': {'ProductName': 'Smartphone', 'Value': '200', 'Weight': '1'}},
                         {'source_row': 1, 'values': {'ProductName': 'Laptop', 'Value': '1500', 'Weight': '5'}},
                         {'source_row': 2, 'values': {'ProductName': 'Headphones', 'Value': '100', 'Weight': '0.5'}},
                         {'source_row': 3, 'values': {'ProductName': 'Camera', 'Value': '800', 'Weight': '2'}},
                         {'source_row': 4, 'values': {'ProductName': 'Smartwatch', 'Value': '250', 'Weight': '0.3'}},
                         {'source_row': 5, 'values': {'ProductName': 'Tablet', 'Value': '600', 'Weight': '1.5'}},
                         {'source_row': 6,
                          'values': {'ProductName': 'Bluetooth Speaker', 'Value': '150', 'Weight': '1'}},
                         {'source_row': 7, 'values': {'ProductName': 'Keyboard', 'Value': '80', 'Weight': '0.8'}},
                         {'source_row': 8, 'values': {'ProductName': 'Mouse', 'Value': '50', 'Weight': '0.2'}},
                         {'source_row': 9, 'values': {'ProductName': 'Monitor', 'Value': '300', 'Weight': '3'}},
                         {'source_row': 10, 'values': {'ProductName': 'Printer', 'Value': '400', 'Weight': '4'}},
                         {'source_row': 11,
                          'values': {'ProductName': 'External Hard Drive', 'Value': '120', 'Weight': '0.5'}},
                         {'source_row': 12, 'values': {'ProductName': 'Router', 'Value': '60', 'Weight': '0.3'}},
                         {'source_row': 13, 'values': {'ProductName': 'Power Bank', 'Value': '40', 'Weight': '0.4'}},
                         {'source_row': 14, 'values': {'ProductName': 'Memory Card', 'Value': '30', 'Weight': '0.05'}},
                         {'source_row': 15,
                          'values': {'ProductName': 'USB Flash Drive', 'Value': '25', 'Weight': '0.02'}},
                         {'source_row': 16,
                          'values': {'ProductName': 'Smart Home Hub', 'Value': '100', 'Weight': '0.6'}},
                         {'source_row': 17, 'values': {'ProductName': 'Gaming Console', 'Value': '500', 'Weight': '4'}},
                         {'source_row': 18,
                          'values': {'ProductName': 'Fitness Tracker', 'Value': '90', 'Weight': '0.2'}},
                         {'source_row': 19, 'values': {'ProductName': 'E-Reader', 'Value': '180', 'Weight': '0.5'}}],
             'returned_rows': 20,
             'role': 'product_data',
             'table_id': 'file_1_view_0'}],
 'validation': {'binding_checks': [{'index_columns': ['ShelfID'],
                                    'key_count': 10,
                                    'parameter': 'capacity',
                                    'status': 'OK',
                                    'table_id': 'file_0_view_0',
                                    'value_column': 'Capacity'},
                                   {'index_columns': ['ProductName'],
                                    'key_count': 20,
                                    'parameter': 'value',
                                    'status': 'OK',
                                    'table_id': 'file_1_view_0',
                                    'value_column': 'Value'},
                                   {'index_columns': ['ProductName'],
                                    'key_count': 20,
                                    'parameter': 'weight',
                                    'status': 'OK',
                                    'table_id': 'file_1_view_0',
                                    'value_column': 'Weight'}],
                'matrix_checks': [],
                'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    data = CSVQA_DATA
    display_table = None
    for t in data['tables']:
        if t['table_id'] == 'file_0_view_0':
            display_table = t
            break
    if display_table is None:
        raise RuntimeError('Display table not found')
    displays = []
    capacity = {}
    for rec in display_table['records']:
        shelf_id = rec['values']['ShelfID']
        displays.append(shelf_id)
        capacity[shelf_id] = float(rec['values']['Capacity'])
    product_table = None
    for t in data['tables']:
        if t['table_id'] == 'file_1_view_0':
            product_table = t
            break
    if product_table is None:
        raise RuntimeError('Product table not found')
    products = []
    value = {}
    weight = {}
    product_source_rows = []
    for rec in product_table['records']:
        pname = rec['values']['ProductName']
        products.append(pname)
        value[pname] = float(rec['values']['Value'])
        weight[pname] = float(rec['values']['Weight'])
        product_source_rows.append(rec['source_row'])
    jstar = None
    for rec in product_table['records']:
        if rec['source_row'] == 0:
            jstar = rec['values']['ProductName']
            break
    if jstar is None:
        raise RuntimeError('First product (source_row=0) not found')
    m = gp.Model('retail_display_allocation')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(displays, products, lb=0, vtype=GRB.INTEGER, name='x')
    m.setObjective(gp.quicksum((value[j] * x[i, j] for i in displays for j in products)), GRB.MAXIMIZE)
    for i in displays:
        m.addConstr(gp.quicksum((weight[j] * x[i, j] for j in products)) <= capacity[i], name=f'capacity_{i}')
    m.addConstr(gp.quicksum((x[i, jstar] for i in displays)) >= 5, name='min_first_product')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()