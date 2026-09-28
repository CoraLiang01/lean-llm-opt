CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'In the context of BigMart Sales, the store needs to allocate various types of products into different '
          'display shelves. Specifically, the store has several shelves, each with a capacity limit provided in '
          '“capacity.csv.” The predefined value and weight of each product can be found in “products.csv.” The '
          'objective is to determine the optimal number of units of each product to place on each shelf to maximize '
          'the total value of the products across all shelves, while ensuring that the total weight of the products on '
          'each shelf does not exceed its capacity. The decision variables x_ij represent the number of units of '
          'product j to be placed on shelf i.The decision variables must be integers.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['ShelfID', 'Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0, 'values': {'Capacity': '500', 'ShelfID': '1'}},
                         {'source_row': 1, 'values': {'Capacity': '700', 'ShelfID': '2'}},
                         {'source_row': 2, 'values': {'Capacity': '600', 'ShelfID': '3'}},
                         {'source_row': 3, 'values': {'Capacity': '800', 'ShelfID': '4'}},
                         {'source_row': 4, 'values': {'Capacity': '550', 'ShelfID': '5'}},
                         {'source_row': 5, 'values': {'Capacity': '900', 'ShelfID': '6'}},
                         {'source_row': 6, 'values': {'Capacity': '650', 'ShelfID': '7'}},
                         {'source_row': 7, 'values': {'Capacity': '750', 'ShelfID': '8'}},
                         {'source_row': 8, 'values': {'Capacity': '820', 'ShelfID': '9'}},
                         {'source_row': 9, 'values': {'Capacity': '570', 'ShelfID': '10'}}],
             'returned_rows': 10,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['ProductName', 'Value', 'Weight'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 20,
             'records': [{'source_row': 0, 'values': {'ProductName': '1', 'Value': '50', 'Weight': '10'}},
                         {'source_row': 1, 'values': {'ProductName': '2', 'Value': '70', 'Weight': '20'}},
                         {'source_row': 2, 'values': {'ProductName': '3', 'Value': '30', 'Weight': '5'}},
                         {'source_row': 3, 'values': {'ProductName': '4', 'Value': '60', 'Weight': '15'}},
                         {'source_row': 4, 'values': {'ProductName': '5', 'Value': '80', 'Weight': '25'}},
                         {'source_row': 5, 'values': {'ProductName': '6', 'Value': '90', 'Weight': '30'}},
                         {'source_row': 6, 'values': {'ProductName': '7', 'Value': '40', 'Weight': '12'}},
                         {'source_row': 7, 'values': {'ProductName': '8', 'Value': '100', 'Weight': '35'}},
                         {'source_row': 8, 'values': {'ProductName': '9', 'Value': '55', 'Weight': '10'}},
                         {'source_row': 9, 'values': {'ProductName': '10', 'Value': '75', 'Weight': '20'}},
                         {'source_row': 10, 'values': {'ProductName': '11', 'Value': '65', 'Weight': '18'}},
                         {'source_row': 11, 'values': {'ProductName': '12', 'Value': '95', 'Weight': '28'}},
                         {'source_row': 12, 'values': {'ProductName': '13', 'Value': '45', 'Weight': '8'}},
                         {'source_row': 13, 'values': {'ProductName': '14', 'Value': '85', 'Weight': '22'}},
                         {'source_row': 14, 'values': {'ProductName': '15', 'Value': '70', 'Weight': '25'}},
                         {'source_row': 15, 'values': {'ProductName': '16', 'Value': '110', 'Weight': '40'}},
                         {'source_row': 16, 'values': {'ProductName': '17', 'Value': '50', 'Weight': '14'}},
                         {'source_row': 17, 'values': {'ProductName': '18', 'Value': '60', 'Weight': '16'}},
                         {'source_row': 18, 'values': {'ProductName': '19', 'Value': '120', 'Weight': '50'}},
                         {'source_row': 19, 'values': {'ProductName': '20', 'Value': '100', 'Weight': '30'}}],
             'returned_rows': 20,
             'role': 'file_1',
             'table_id': 'file_1_view_0'}],
 'validation': {'fallback_reason': "Relationship references an unknown table_id: {'type': 'matrix', 'matrix_table_id': "
                                   "'file_2_view_0', 'row_id_column': 'ShelfID', 'row_axis': {'table_id': "
                                   "'file_0_view_0', 'id_column': 'ShelfID'}, 'column_axis': {'table_id': "
                                   "'file_1_view_0', 'id_column': 'ProductName'}}",
                'planner_errors': ["Relationship references an unknown table_id: {'type': 'matrix', 'matrix_table_id': "
                                   "'file_2_view_0', 'row_id_column': 'ShelfID', 'row_axis': {'table_id': "
                                   "'file_0_view_0', 'id_column': 'ShelfID'}, 'column_axis': {'table_id': "
                                   "'file_1_view_0', 'id_column': 'ProductName'}}"],
                'status': 'FALLBACK_FULL_DATA'}}
import gurobipy as gp
from gurobipy import GRB
CSVQA_DATA = {'ignored_file_indices': [], 'query': 'In the context of BigMart Sales, the store needs to allocate various types of products into different display shelves. Specifically, the store has several shelves, each with a capacity limit provided in “capacity.csv.” The predefined value and weight of each product can be found in “products.csv.” The objective is to determine the optimal number of units of each product to place on each shelf to maximize the total value of the products across all shelves, while ensuring that the total weight of the products on each shelf does not exceed its capacity. The decision variables x_ij represent the number of units of product j to be placed on shelf i.The decision variables must be integers.', 'relationships': [], 'route': 'NRM', 'tables': [{'columns': ['ShelfID', 'Capacity'], 'file_index': 0, 'file_name': 'capacity.csv', 'filters': {'conditions': [], 'logic': 'and'}, 'original_rows': 10, 'records': [{'source_row': 0, 'values': {'Capacity': '500', 'ShelfID': '1'}}, {'source_row': 1, 'values': {'Capacity': '700', 'ShelfID': '2'}}, {'source_row': 2, 'values': {'Capacity': '600', 'ShelfID': '3'}}, {'source_row': 3, 'values': {'Capacity': '800', 'ShelfID': '4'}}, {'source_row': 4, 'values': {'Capacity': '550', 'ShelfID': '5'}}, {'source_row': 5, 'values': {'Capacity': '900', 'ShelfID': '6'}}, {'source_row': 6, 'values': {'Capacity': '650', 'ShelfID': '7'}}, {'source_row': 7, 'values': {'Capacity': '750', 'ShelfID': '8'}}, {'source_row': 8, 'values': {'Capacity': '820', 'ShelfID': '9'}}, {'source_row': 9, 'values': {'Capacity': '570', 'ShelfID': '10'}}], 'returned_rows': 10, 'role': 'file_0', 'table_id': 'file_0_view_0'}, {'columns': ['ProductName', 'Value', 'Weight'], 'file_index': 1, 'file_name': 'products.csv', 'filters': {'conditions': [], 'logic': 'and'}, 'original_rows': 20, 'records': [{'source_row': 0, 'values': {'ProductName': '1', 'Value': '50', 'Weight': '10'}}, {'source_row': 1, 'values': {'ProductName': '2', 'Value': '70', 'Weight': '20'}}, {'source_row': 2, 'values': {'ProductName': '3', 'Value': '30', 'Weight': '5'}}, {'source_row': 3, 'values': {'ProductName': '4', 'Value': '60', 'Weight': '15'}}, {'source_row': 4, 'values': {'ProductName': '5', 'Value': '80', 'Weight': '25'}}, {'source_row': 5, 'values': {'ProductName': '6', 'Value': '90', 'Weight': '30'}}, {'source_row': 6, 'values': {'ProductName': '7', 'Value': '40', 'Weight': '12'}}, {'source_row': 7, 'values': {'ProductName': '8', 'Value': '100', 'Weight': '35'}}, {'source_row': 8, 'values': {'ProductName': '9', 'Value': '55', 'Weight': '10'}}, {'source_row': 9, 'values': {'ProductName': '10', 'Value': '75', 'Weight': '20'}}, {'source_row': 10, 'values': {'ProductName': '11', 'Value': '65', 'Weight': '18'}}, {'source_row': 11, 'values': {'ProductName': '12', 'Value': '95', 'Weight': '28'}}, {'source_row': 12, 'values': {'ProductName': '13', 'Value': '45', 'Weight': '8'}}, {'source_row': 13, 'values': {'ProductName': '14', 'Value': '85', 'Weight': '22'}}, {'source_row': 14, 'values': {'ProductName': '15', 'Value': '70', 'Weight': '25'}}, {'source_row': 15, 'values': {'ProductName': '16', 'Value': '110', 'Weight': '40'}}, {'source_row': 16, 'values': {'ProductName': '17', 'Value': '50', 'Weight': '14'}}, {'source_row': 17, 'values': {'ProductName': '18', 'Value': '60', 'Weight': '16'}}, {'source_row': 18, 'values': {'ProductName': '19', 'Value': '120', 'Weight': '50'}}, {'source_row': 19, 'values': {'ProductName': '20', 'Value': '100', 'Weight': '30'}}], 'returned_rows': 20, 'role': 'file_1', 'table_id': 'file_1_view_0'}], 'validation': {'fallback_reason': "Relationship references an unknown table_id: {'type': 'matrix', 'matrix_table_id': 'file_2_view_0', 'row_id_column': 'ShelfID', 'row_axis': {'table_id': 'file_0_view_0', 'id_column': 'ShelfID'}, 'column_axis': {'table_id': 'file_1_view_0', 'id_column': 'ProductName'}}", 'planner_errors': ["Relationship references an unknown table_id: {'type': 'matrix', 'matrix_table_id': 'file_2_view_0', 'row_id_column': 'ShelfID', 'row_axis': {'table_id': 'file_0_view_0', 'id_column': 'ShelfID'}, 'column_axis': {'table_id': 'file_1_view_0', 'id_column': 'ProductName'}}"], 'status': 'FALLBACK_FULL_DATA'}}
shelf_table = None
for t in CSVQA_DATA['tables']:
    if t['table_id'] == 'file_0_view_0':
        shelf_table = t
        break
if shelf_table is None:
    raise ValueError('Shelf table file_0_view_0 not found in data.')
S = []
C = {}
for rec in shelf_table['records']:
    shelf_id = rec['values']['ShelfID']
    cap = rec['values']['Capacity']
    S.append(shelf_id)
    C[shelf_id] = int(cap)
product_table = None
for t in CSVQA_DATA['tables']:
    if t['table_id'] == 'file_1_view_0':
        product_table = t
        break
if product_table is None:
    raise ValueError('Product table file_1_view_0 not found in data.')
P = []
v = {}
w = {}
for rec in product_table['records']:
    pname = rec['values']['ProductName']
    val = rec['values']['Value']
    wei = rec['values']['Weight']
    P.append(pname)
    v[pname] = int(val)
    w[pname] = int(wei)
if set(C.keys()) != set(S):
    raise ValueError('Shelf capacity data missing for some shelves.')
if set(v.keys()) != set(P) or set(w.keys()) != set(P):
    raise ValueError('Product value/weight data missing for some products.')
m = gp.Model('BigMart_Shelf_Allocation')
x = m.addVars(S, P, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((v[p] * x[s, p] for s in S for p in P)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((w[p] * x[s, p] for p in P)) <= C[s] for s in S), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for s in S:
        for p in P:
            var = x[s, p]
            print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')