CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'In retail, shops need to allocate various types of products to different displays. The capacity limit of '
          'each display is provided in "capacity.csv", and the value and weight of each product are provided in '
          '"products.csv". The objective is to determine the optimal number of each product to place on each display '
          'so as to maximize the total value of all products placed across the displays, while ensuring that the total '
          'weight of the products on each display does not exceed its capacity. In addition, the total quantity of the '
          'first product placed across all displays must be at least 5. The decision variable x_{ij} represents the '
          'number of units of product j placed on display i.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'columns': ['ShelfID', 'ArchiveRevisionCount', 'Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0, 'values': {'ArchiveRevisionCount': '16', 'Capacity': '5', 'ShelfID': '1'}},
                         {'source_row': 1, 'values': {'ArchiveRevisionCount': '17', 'Capacity': '7', 'ShelfID': '2'}},
                         {'source_row': 2, 'values': {'ArchiveRevisionCount': '18', 'Capacity': '6', 'ShelfID': '3'}},
                         {'source_row': 3, 'values': {'ArchiveRevisionCount': '12', 'Capacity': '8', 'ShelfID': '4'}},
                         {'source_row': 4, 'values': {'ArchiveRevisionCount': '7', 'Capacity': '5.5', 'ShelfID': '5'}},
                         {'source_row': 5, 'values': {'ArchiveRevisionCount': '1', 'Capacity': '9', 'ShelfID': '6'}},
                         {'source_row': 6, 'values': {'ArchiveRevisionCount': '27', 'Capacity': '6.5', 'ShelfID': '7'}},
                         {'source_row': 7, 'values': {'ArchiveRevisionCount': '20', 'Capacity': '7.5', 'ShelfID': '8'}},
                         {'source_row': 8, 'values': {'ArchiveRevisionCount': '22', 'Capacity': '8.2', 'ShelfID': '9'}},
                         {'source_row': 9,
                          'values': {'ArchiveRevisionCount': '28', 'Capacity': '5.7', 'ShelfID': '10'}}],
             'returned_rows': 10,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['ArchiveRevisionCount', 'ArchiveFolder', 'ProductName', 'Value', 'Weight'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 20,
             'records': [{'source_row': 0,
                          'values': {'ArchiveFolder': 'Folder_C',
                                     'ArchiveRevisionCount': '15',
                                     'ProductName': 'Smartphone',
                                     'Value': '200',
                                     'Weight': '1'}},
                         {'source_row': 1,
                          'values': {'ArchiveFolder': 'Folder_B',
                                     'ArchiveRevisionCount': '15',
                                     'ProductName': 'Laptop',
                                     'Value': '1500',
                                     'Weight': '5'}},
                         {'source_row': 2,
                          'values': {'ArchiveFolder': 'Folder_C',
                                     'ArchiveRevisionCount': '5',
                                     'ProductName': 'Headphones',
                                     'Value': '100',
                                     'Weight': '0.5'}},
                         {'source_row': 3,
                          'values': {'ArchiveFolder': 'Folder_B',
                                     'ArchiveRevisionCount': '8',
                                     'ProductName': 'Camera',
                                     'Value': '800',
                                     'Weight': '2'}},
                         {'source_row': 4,
                          'values': {'ArchiveFolder': 'Folder_A',
                                     'ArchiveRevisionCount': '13',
                                     'ProductName': 'Smartwatch',
                                     'Value': '250',
                                     'Weight': '0.3'}},
                         {'source_row': 5,
                          'values': {'ArchiveFolder': 'Folder_C',
                                     'ArchiveRevisionCount': '28',
                                     'ProductName': 'Tablet',
                                     'Value': '600',
                                     'Weight': '1.5'}},
                         {'source_row': 6,
                          'values': {'ArchiveFolder': 'Folder_B',
                                     'ArchiveRevisionCount': '23',
                                     'ProductName': 'Bluetooth Speaker',
                                     'Value': '150',
                                     'Weight': '1'}},
                         {'source_row': 7,
                          'values': {'ArchiveFolder': 'Folder_B',
                                     'ArchiveRevisionCount': '4',
                                     'ProductName': 'Keyboard',
                                     'Value': '80',
                                     'Weight': '0.8'}},
                         {'source_row': 8,
                          'values': {'ArchiveFolder': 'Folder_B',
                                     'ArchiveRevisionCount': '22',
                                     'ProductName': 'Mouse',
                                     'Value': '50',
                                     'Weight': '0.2'}},
                         {'source_row': 9,
                          'values': {'ArchiveFolder': 'Folder_C',
                                     'ArchiveRevisionCount': '16',
                                     'ProductName': 'Monitor',
                                     'Value': '300',
                                     'Weight': '3'}},
                         {'source_row': 10,
                          'values': {'ArchiveFolder': 'Folder_B',
                                     'ArchiveRevisionCount': '29',
                                     'ProductName': 'Printer',
                                     'Value': '400',
                                     'Weight': '4'}},
                         {'source_row': 11,
                          'values': {'ArchiveFolder': 'Folder_B',
                                     'ArchiveRevisionCount': '12',
                                     'ProductName': 'External Hard Drive',
                                     'Value': '120',
                                     'Weight': '0.5'}},
                         {'source_row': 12,
                          'values': {'ArchiveFolder': 'Folder_A',
                                     'ArchiveRevisionCount': '19',
                                     'ProductName': 'Router',
                                     'Value': '60',
                                     'Weight': '0.3'}},
                         {'source_row': 13,
                          'values': {'ArchiveFolder': 'Folder_C',
                                     'ArchiveRevisionCount': '14',
                                     'ProductName': 'Power Bank',
                                     'Value': '40',
                                     'Weight': '0.4'}},
                         {'source_row': 14,
                          'values': {'ArchiveFolder': 'Folder_B',
                                     'ArchiveRevisionCount': '5',
                                     'ProductName': 'Memory Card',
                                     'Value': '30',
                                     'Weight': '0.05'}},
                         {'source_row': 15,
                          'values': {'ArchiveFolder': 'Folder_C',
                                     'ArchiveRevisionCount': '29',
                                     'ProductName': 'USB Flash Drive',
                                     'Value': '25',
                                     'Weight': '0.02'}},
                         {'source_row': 16,
                          'values': {'ArchiveFolder': 'Folder_A',
                                     'ArchiveRevisionCount': '20',
                                     'ProductName': 'Smart Home Hub',
                                     'Value': '100',
                                     'Weight': '0.6'}},
                         {'source_row': 17,
                          'values': {'ArchiveFolder': 'Folder_A',
                                     'ArchiveRevisionCount': '30',
                                     'ProductName': 'Gaming Console',
                                     'Value': '500',
                                     'Weight': '4'}},
                         {'source_row': 18,
                          'values': {'ArchiveFolder': 'Folder_A',
                                     'ArchiveRevisionCount': '16',
                                     'ProductName': 'Fitness Tracker',
                                     'Value': '90',
                                     'Weight': '0.2'}},
                         {'source_row': 19,
                          'values': {'ArchiveFolder': 'Folder_A',
                                     'ArchiveRevisionCount': '18',
                                     'ProductName': 'E-Reader',
                                     'Value': '180',
                                     'Weight': '0.5'}}],
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
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem(CSVQA_FRAMES):
    frame0 = CSVQA_FRAMES['file_0_view_0']
    displays = []
    capacity = {}
    for (source_row, row) in frame0.iterrows():
        shelf_id = row['ShelfID']
        displays.append(shelf_id)
        try:
            capacity[shelf_id] = float(row['Capacity'])
        except Exception:
            raise ValueError(f"Invalid Capacity for ShelfID {shelf_id}: {row['Capacity']}")
    frame1 = CSVQA_FRAMES['file_1_view_0']
    products = []
    value = {}
    weight = {}
    for (source_row, row) in frame1.iterrows():
        product = row['ProductName']
        products.append(product)
        try:
            value[product] = float(row['Value'])
        except Exception:
            raise ValueError(f"Invalid Value for ProductName {product}: {row['Value']}")
        try:
            weight[product] = float(row['Weight'])
        except Exception:
            raise ValueError(f"Invalid Weight for ProductName {product}: {row['Weight']}")
    first_product = frame1.iloc[0]['ProductName']
    m = gp.Model('retail_display_allocation')
    quantity_vars = m.addVars(displays, products, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((value[j] * quantity_vars[i, j] for i in displays for j in products)), GRB.MAXIMIZE)
    m.addConstrs((gp.quicksum((weight[j] * quantity_vars[i, j] for j in products)) <= capacity[i] for i in displays), name='')
    m.addConstr(gp.quicksum((quantity_vars[i, first_product] for i in displays)) >= 5, name='min_first_product')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem(CSVQA_FRAMES)