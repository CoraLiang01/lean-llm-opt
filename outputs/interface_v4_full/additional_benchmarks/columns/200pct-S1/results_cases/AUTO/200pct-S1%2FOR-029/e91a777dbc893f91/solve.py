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
 'tables': [{'columns': ['ArchiveAccessCount',
                         'ShelfID',
                         'ArchiveRevisionCount',
                         'Capacity',
                         'ArchiveAttachmentCount',
                         'ArchivePageCount'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0,
                          'values': {'ArchiveAccessCount': '14',
                                     'ArchiveAttachmentCount': '7',
                                     'ArchivePageCount': '23',
                                     'ArchiveRevisionCount': '16',
                                     'Capacity': '5',
                                     'ShelfID': '1'}},
                         {'source_row': 1,
                          'values': {'ArchiveAccessCount': '16',
                                     'ArchiveAttachmentCount': '23',
                                     'ArchivePageCount': '20',
                                     'ArchiveRevisionCount': '17',
                                     'Capacity': '7',
                                     'ShelfID': '2'}},
                         {'source_row': 2,
                          'values': {'ArchiveAccessCount': '16',
                                     'ArchiveAttachmentCount': '15',
                                     'ArchivePageCount': '28',
                                     'ArchiveRevisionCount': '18',
                                     'Capacity': '6',
                                     'ShelfID': '3'}},
                         {'source_row': 3,
                          'values': {'ArchiveAccessCount': '14',
                                     'ArchiveAttachmentCount': '14',
                                     'ArchivePageCount': '18',
                                     'ArchiveRevisionCount': '12',
                                     'Capacity': '8',
                                     'ShelfID': '4'}},
                         {'source_row': 4,
                          'values': {'ArchiveAccessCount': '2',
                                     'ArchiveAttachmentCount': '17',
                                     'ArchivePageCount': '19',
                                     'ArchiveRevisionCount': '7',
                                     'Capacity': '5.5',
                                     'ShelfID': '5'}},
                         {'source_row': 5,
                          'values': {'ArchiveAccessCount': '18',
                                     'ArchiveAttachmentCount': '4',
                                     'ArchivePageCount': '29',
                                     'ArchiveRevisionCount': '1',
                                     'Capacity': '9',
                                     'ShelfID': '6'}},
                         {'source_row': 6,
                          'values': {'ArchiveAccessCount': '12',
                                     'ArchiveAttachmentCount': '26',
                                     'ArchivePageCount': '9',
                                     'ArchiveRevisionCount': '27',
                                     'Capacity': '6.5',
                                     'ShelfID': '7'}},
                         {'source_row': 7,
                          'values': {'ArchiveAccessCount': '30',
                                     'ArchiveAttachmentCount': '11',
                                     'ArchivePageCount': '14',
                                     'ArchiveRevisionCount': '20',
                                     'Capacity': '7.5',
                                     'ShelfID': '8'}},
                         {'source_row': 8,
                          'values': {'ArchiveAccessCount': '9',
                                     'ArchiveAttachmentCount': '15',
                                     'ArchivePageCount': '26',
                                     'ArchiveRevisionCount': '22',
                                     'Capacity': '8.2',
                                     'ShelfID': '9'}},
                         {'source_row': 9,
                          'values': {'ArchiveAccessCount': '8',
                                     'ArchiveAttachmentCount': '30',
                                     'ArchivePageCount': '12',
                                     'ArchiveRevisionCount': '28',
                                     'Capacity': '5.7',
                                     'ShelfID': '10'}}],
             'returned_rows': 10,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['ArchivePageCount',
                         'ArchiveRevisionCount',
                         'ArchiveFolder',
                         'ArchiveAttachmentCount',
                         'ProductName',
                         'ArchiveReviewDesk',
                         'Value',
                         'ArchiveAccessRoute',
                         'Weight'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 20,
             'records': [{'source_row': 0,
                          'values': {'ArchiveAccessRoute': 'CatalogIndex',
                                     'ArchiveAttachmentCount': '15',
                                     'ArchiveFolder': 'Folder_C',
                                     'ArchivePageCount': '4',
                                     'ArchiveReviewDesk': 'Desk_B',
                                     'ArchiveRevisionCount': '15',
                                     'ProductName': 'Smartphone',
                                     'Value': '200',
                                     'Weight': '1'}},
                         {'source_row': 1,
                          'values': {'ArchiveAccessRoute': 'LocalIndex',
                                     'ArchiveAttachmentCount': '9',
                                     'ArchiveFolder': 'Folder_B',
                                     'ArchivePageCount': '7',
                                     'ArchiveReviewDesk': 'Desk_A',
                                     'ArchiveRevisionCount': '15',
                                     'ProductName': 'Laptop',
                                     'Value': '1500',
                                     'Weight': '5'}},
                         {'source_row': 2,
                          'values': {'ArchiveAccessRoute': 'CatalogIndex',
                                     'ArchiveAttachmentCount': '12',
                                     'ArchiveFolder': 'Folder_C',
                                     'ArchivePageCount': '25',
                                     'ArchiveReviewDesk': 'Desk_C',
                                     'ArchiveRevisionCount': '5',
                                     'ProductName': 'Headphones',
                                     'Value': '100',
                                     'Weight': '0.5'}},
                         {'source_row': 3,
                          'values': {'ArchiveAccessRoute': 'Portal',
                                     'ArchiveAttachmentCount': '7',
                                     'ArchiveFolder': 'Folder_B',
                                     'ArchivePageCount': '25',
                                     'ArchiveReviewDesk': 'Desk_B',
                                     'ArchiveRevisionCount': '8',
                                     'ProductName': 'Camera',
                                     'Value': '800',
                                     'Weight': '2'}},
                         {'source_row': 4,
                          'values': {'ArchiveAccessRoute': 'LocalIndex',
                                     'ArchiveAttachmentCount': '18',
                                     'ArchiveFolder': 'Folder_A',
                                     'ArchivePageCount': '30',
                                     'ArchiveReviewDesk': 'Desk_C',
                                     'ArchiveRevisionCount': '13',
                                     'ProductName': 'Smartwatch',
                                     'Value': '250',
                                     'Weight': '0.3'}},
                         {'source_row': 5,
                          'values': {'ArchiveAccessRoute': 'CatalogIndex',
                                     'ArchiveAttachmentCount': '27',
                                     'ArchiveFolder': 'Folder_C',
                                     'ArchivePageCount': '16',
                                     'ArchiveReviewDesk': 'Desk_A',
                                     'ArchiveRevisionCount': '28',
                                     'ProductName': 'Tablet',
                                     'Value': '600',
                                     'Weight': '1.5'}},
                         {'source_row': 6,
                          'values': {'ArchiveAccessRoute': 'LocalIndex',
                                     'ArchiveAttachmentCount': '12',
                                     'ArchiveFolder': 'Folder_B',
                                     'ArchivePageCount': '24',
                                     'ArchiveReviewDesk': 'Desk_A',
                                     'ArchiveRevisionCount': '23',
                                     'ProductName': 'Bluetooth Speaker',
                                     'Value': '150',
                                     'Weight': '1'}},
                         {'source_row': 7,
                          'values': {'ArchiveAccessRoute': 'LocalIndex',
                                     'ArchiveAttachmentCount': '28',
                                     'ArchiveFolder': 'Folder_B',
                                     'ArchivePageCount': '10',
                                     'ArchiveReviewDesk': 'Desk_B',
                                     'ArchiveRevisionCount': '4',
                                     'ProductName': 'Keyboard',
                                     'Value': '80',
                                     'Weight': '0.8'}},
                         {'source_row': 8,
                          'values': {'ArchiveAccessRoute': 'Portal',
                                     'ArchiveAttachmentCount': '4',
                                     'ArchiveFolder': 'Folder_B',
                                     'ArchivePageCount': '23',
                                     'ArchiveReviewDesk': 'Desk_B',
                                     'ArchiveRevisionCount': '22',
                                     'ProductName': 'Mouse',
                                     'Value': '50',
                                     'Weight': '0.2'}},
                         {'source_row': 9,
                          'values': {'ArchiveAccessRoute': 'CatalogIndex',
                                     'ArchiveAttachmentCount': '10',
                                     'ArchiveFolder': 'Folder_C',
                                     'ArchivePageCount': '9',
                                     'ArchiveReviewDesk': 'Desk_C',
                                     'ArchiveRevisionCount': '16',
                                     'ProductName': 'Monitor',
                                     'Value': '300',
                                     'Weight': '3'}},
                         {'source_row': 10,
                          'values': {'ArchiveAccessRoute': 'CatalogIndex',
                                     'ArchiveAttachmentCount': '22',
                                     'ArchiveFolder': 'Folder_B',
                                     'ArchivePageCount': '11',
                                     'ArchiveReviewDesk': 'Desk_C',
                                     'ArchiveRevisionCount': '29',
                                     'ProductName': 'Printer',
                                     'Value': '400',
                                     'Weight': '4'}},
                         {'source_row': 11,
                          'values': {'ArchiveAccessRoute': 'Portal',
                                     'ArchiveAttachmentCount': '10',
                                     'ArchiveFolder': 'Folder_B',
                                     'ArchivePageCount': '22',
                                     'ArchiveReviewDesk': 'Desk_A',
                                     'ArchiveRevisionCount': '12',
                                     'ProductName': 'External Hard Drive',
                                     'Value': '120',
                                     'Weight': '0.5'}},
                         {'source_row': 12,
                          'values': {'ArchiveAccessRoute': 'LocalIndex',
                                     'ArchiveAttachmentCount': '19',
                                     'ArchiveFolder': 'Folder_A',
                                     'ArchivePageCount': '28',
                                     'ArchiveReviewDesk': 'Desk_C',
                                     'ArchiveRevisionCount': '19',
                                     'ProductName': 'Router',
                                     'Value': '60',
                                     'Weight': '0.3'}},
                         {'source_row': 13,
                          'values': {'ArchiveAccessRoute': 'Portal',
                                     'ArchiveAttachmentCount': '23',
                                     'ArchiveFolder': 'Folder_C',
                                     'ArchivePageCount': '5',
                                     'ArchiveReviewDesk': 'Desk_B',
                                     'ArchiveRevisionCount': '14',
                                     'ProductName': 'Power Bank',
                                     'Value': '40',
                                     'Weight': '0.4'}},
                         {'source_row': 14,
                          'values': {'ArchiveAccessRoute': 'Portal',
                                     'ArchiveAttachmentCount': '12',
                                     'ArchiveFolder': 'Folder_B',
                                     'ArchivePageCount': '22',
                                     'ArchiveReviewDesk': 'Desk_C',
                                     'ArchiveRevisionCount': '5',
                                     'ProductName': 'Memory Card',
                                     'Value': '30',
                                     'Weight': '0.05'}},
                         {'source_row': 15,
                          'values': {'ArchiveAccessRoute': 'LocalIndex',
                                     'ArchiveAttachmentCount': '29',
                                     'ArchiveFolder': 'Folder_C',
                                     'ArchivePageCount': '7',
                                     'ArchiveReviewDesk': 'Desk_A',
                                     'ArchiveRevisionCount': '29',
                                     'ProductName': 'USB Flash Drive',
                                     'Value': '25',
                                     'Weight': '0.02'}},
                         {'source_row': 16,
                          'values': {'ArchiveAccessRoute': 'CatalogIndex',
                                     'ArchiveAttachmentCount': '11',
                                     'ArchiveFolder': 'Folder_A',
                                     'ArchivePageCount': '29',
                                     'ArchiveReviewDesk': 'Desk_C',
                                     'ArchiveRevisionCount': '20',
                                     'ProductName': 'Smart Home Hub',
                                     'Value': '100',
                                     'Weight': '0.6'}},
                         {'source_row': 17,
                          'values': {'ArchiveAccessRoute': 'CatalogIndex',
                                     'ArchiveAttachmentCount': '24',
                                     'ArchiveFolder': 'Folder_A',
                                     'ArchivePageCount': '8',
                                     'ArchiveReviewDesk': 'Desk_A',
                                     'ArchiveRevisionCount': '30',
                                     'ProductName': 'Gaming Console',
                                     'Value': '500',
                                     'Weight': '4'}},
                         {'source_row': 18,
                          'values': {'ArchiveAccessRoute': 'LocalIndex',
                                     'ArchiveAttachmentCount': '5',
                                     'ArchiveFolder': 'Folder_A',
                                     'ArchivePageCount': '19',
                                     'ArchiveReviewDesk': 'Desk_B',
                                     'ArchiveRevisionCount': '16',
                                     'ProductName': 'Fitness Tracker',
                                     'Value': '90',
                                     'Weight': '0.2'}},
                         {'source_row': 19,
                          'values': {'ArchiveAccessRoute': 'CatalogIndex',
                                     'ArchiveAttachmentCount': '17',
                                     'ArchiveFolder': 'Folder_A',
                                     'ArchivePageCount': '21',
                                     'ArchiveReviewDesk': 'Desk_B',
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
import gurobipy as gp
from gurobipy import GRB

def solve_problem(CSVQA_DATA):
    displays_table = None
    for t in CSVQA_DATA['tables']:
        if t['table_id'] == 'file_0_view_0':
            displays_table = t
            break
    if displays_table is None:
        raise RuntimeError('Missing displays table file_0_view_0')
    I = []
    c = {}
    for rec in displays_table['records']:
        shelf_id = rec['values']['ShelfID']
        I.append(shelf_id)
        cap = rec['values']['Capacity']
        try:
            c[shelf_id] = float(cap)
        except Exception:
            raise ValueError(f'Invalid capacity for ShelfID {shelf_id}: {cap}')
    products_table = None
    for t in CSVQA_DATA['tables']:
        if t['table_id'] == 'file_1_view_0':
            products_table = t
            break
    if products_table is None:
        raise RuntimeError('Missing products table file_1_view_0')
    J = []
    v = {}
    w = {}
    jstar = None
    for rec in products_table['records']:
        pname = rec['values']['ProductName']
        J.append(pname)
        val = rec['values']['Value']
        wei = rec['values']['Weight']
        try:
            v[pname] = float(val)
            w[pname] = float(wei)
        except Exception:
            raise ValueError(f'Invalid value/weight for ProductName {pname}: Value={val}, Weight={wei}')
        if rec['source_row'] == 0:
            jstar = pname
    if jstar is None:
        raise RuntimeError('Could not determine j* (first product)')
    if set(c.keys()) != set(I):
        raise RuntimeError('Capacity keys do not match display indices')
    if set(v.keys()) != set(J) or set(w.keys()) != set(J):
        raise RuntimeError('Value/weight keys do not match product indices')
    m = gp.Model('retail_display_allocation')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(I, J, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v[j] * x[i, j] for i in I for j in J)), GRB.MAXIMIZE)
    m.addConstrs((gp.quicksum((w[j] * x[i, j] for j in J)) <= c[i] for i in I), name='')
    m.addConstr(gp.quicksum((x[i, jstar] for i in I)) >= 5, name='min_first_product')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem(CSVQA_DATA)