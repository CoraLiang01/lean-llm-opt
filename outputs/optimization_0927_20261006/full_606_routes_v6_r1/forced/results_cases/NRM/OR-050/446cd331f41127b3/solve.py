CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'In retail, shops need to allocate various types of products to different displays. The capacity limit of '
          'each display is provided in "capacity.csv", and the value and weight of each product are provided in '
          '"products.csv". The objective is to determine the optimal number of each product to place on each display '
          'so as to maximize the total value of all products placed across the displays, while ensuring that the total '
          'weight of the products on each display does not exceed its capacity. In addition, the total quantity of the '
          'first product placed across all displays must be at least 5. The decision variable x_{ij} represents the '
          'number of units of product j placed on display i.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['ShelfID', 'Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0, 'values': {'Capacity': '5.0', 'ShelfID': '1'}},
                         {'source_row': 1, 'values': {'Capacity': '7.0', 'ShelfID': '2'}},
                         {'source_row': 2, 'values': {'Capacity': '6.0', 'ShelfID': '3'}},
                         {'source_row': 3, 'values': {'Capacity': '8.0', 'ShelfID': '4'}},
                         {'source_row': 4, 'values': {'Capacity': '5.5', 'ShelfID': '5'}},
                         {'source_row': 5, 'values': {'Capacity': '9.0', 'ShelfID': '6'}},
                         {'source_row': 6, 'values': {'Capacity': '6.5', 'ShelfID': '7'}},
                         {'source_row': 7, 'values': {'Capacity': '7.5', 'ShelfID': '8'}},
                         {'source_row': 8, 'values': {'Capacity': '8.2', 'ShelfID': '9'}},
                         {'source_row': 9, 'values': {'Capacity': '5.7', 'ShelfID': '10'}}],
             'returned_rows': 10,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['ProductName', 'Value', 'Weight'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 20,
             'records': [{'source_row': 0, 'values': {'ProductName': 'Smartphone', 'Value': '200', 'Weight': '1.0'}},
                         {'source_row': 1, 'values': {'ProductName': 'Laptop', 'Value': '1500', 'Weight': '5.0'}},
                         {'source_row': 2, 'values': {'ProductName': 'Headphones', 'Value': '100', 'Weight': '0.5'}},
                         {'source_row': 3, 'values': {'ProductName': 'Camera', 'Value': '800', 'Weight': '2.0'}},
                         {'source_row': 4, 'values': {'ProductName': 'Smartwatch', 'Value': '250', 'Weight': '0.3'}},
                         {'source_row': 5, 'values': {'ProductName': 'Tablet', 'Value': '600', 'Weight': '1.5'}},
                         {'source_row': 6,
                          'values': {'ProductName': 'Bluetooth Speaker', 'Value': '150', 'Weight': '1.0'}},
                         {'source_row': 7, 'values': {'ProductName': 'Keyboard', 'Value': '80', 'Weight': '0.8'}},
                         {'source_row': 8, 'values': {'ProductName': 'Mouse', 'Value': '50', 'Weight': '0.2'}},
                         {'source_row': 9, 'values': {'ProductName': 'Monitor', 'Value': '300', 'Weight': '3.0'}},
                         {'source_row': 10, 'values': {'ProductName': 'Printer', 'Value': '400', 'Weight': '4.0'}},
                         {'source_row': 11,
                          'values': {'ProductName': 'External Hard Drive', 'Value': '120', 'Weight': '0.5'}},
                         {'source_row': 12, 'values': {'ProductName': 'Router', 'Value': '60', 'Weight': '0.3'}},
                         {'source_row': 13, 'values': {'ProductName': 'Power Bank', 'Value': '40', 'Weight': '0.4'}},
                         {'source_row': 14, 'values': {'ProductName': 'Memory Card', 'Value': '30', 'Weight': '0.05'}},
                         {'source_row': 15,
                          'values': {'ProductName': 'USB Flash Drive', 'Value': '25', 'Weight': '0.02'}},
                         {'source_row': 16,
                          'values': {'ProductName': 'Smart Home Hub', 'Value': '100', 'Weight': '0.6'}},
                         {'source_row': 17,
                          'values': {'ProductName': 'Gaming Console', 'Value': '500', 'Weight': '4.0'}},
                         {'source_row': 18,
                          'values': {'ProductName': 'Fitness Tracker', 'Value': '90', 'Weight': '0.2'}},
                         {'source_row': 19, 'values': {'ProductName': 'E-Reader', 'Value': '180', 'Weight': '0.5'}}],
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
capacity_table = [rec['values'] for rec in CSVQA_DATA['tables'][0]['records']]
products_table = [rec['values'] for rec in CSVQA_DATA['tables'][1]['records']]
I = [row['ShelfID'] for row in capacity_table]
J = [row['ProductName'] for row in products_table]
C = {row['ShelfID']: float(row['Capacity']) for row in capacity_table}
v = {row['ProductName']: float(row['Value']) for row in products_table}
w = {row['ProductName']: float(row['Weight']) for row in products_table}
first_product = products_table[0]['ProductName']
if set(C.keys()) != set(I):
    raise ValueError('Mismatch in display identifiers between capacity data and index set I.')
if set(v.keys()) != set(J) or set(w.keys()) != set(J):
    raise ValueError('Mismatch in product identifiers between product data and index set J.')
m = gp.Model('Retail_Display_Allocation')
x_vars = m.addVars(I, J, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((v[j] * x_vars[i, j] for i in I for j in J)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((w[j] * x_vars[i, j] for j in J)) <= C[i] for i in I), name='')
m.addConstr(gp.quicksum((x_vars[i, first_product] for i in I)) >= 5, name='min_first_product')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')