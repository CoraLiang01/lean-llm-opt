CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'In the context of retail sales, the store needs to allocate various types of products into different '
          'display shelves. Specifically, the store has several display shelves, each with a capacity limit provided '
          'in ‚Äúcapacity.csv.‚Äù The predefined value and weight of each product can be found in ‚Äúproducts.csv.‚Äù '
          'The objective is to determine the optimal number of units of each product to place on each display shelf to '
          'maximize the total value of the products across all shelves while ensuring that the total weight of the '
          'products on each shelf does not exceed its capacity. The decision variablesx_ijrepresent the number of '
          'units of product j to be placed on shelf i.The decision variables must be integers.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'columns': ['shelf_cleaning_minutes_last_month', 'ShelfID', 'store_weekly_visitor_count', 'Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0,
                          'values': {'Capacity': '5.0',
                                     'ShelfID': '1',
                                     'shelf_cleaning_minutes_last_month': '60',
                                     'store_weekly_visitor_count': '1820'}},
                         {'source_row': 1,
                          'values': {'Capacity': '7.0',
                                     'ShelfID': '2',
                                     'shelf_cleaning_minutes_last_month': '120',
                                     'store_weekly_visitor_count': '1820'}},
                         {'source_row': 2,
                          'values': {'Capacity': '6.0',
                                     'ShelfID': '3',
                                     'shelf_cleaning_minutes_last_month': '60',
                                     'store_weekly_visitor_count': '850'}},
                         {'source_row': 3,
                          'values': {'Capacity': '8.0',
                                     'ShelfID': '4',
                                     'shelf_cleaning_minutes_last_month': '60',
                                     'store_weekly_visitor_count': '850'}},
                         {'source_row': 4,
                          'values': {'Capacity': '5.5',
                                     'ShelfID': '5',
                                     'shelf_cleaning_minutes_last_month': '45',
                                     'store_weekly_visitor_count': '1460'}},
                         {'source_row': 5,
                          'values': {'Capacity': '9.0',
                                     'ShelfID': '6',
                                     'shelf_cleaning_minutes_last_month': '60',
                                     'store_weekly_visitor_count': '850'}},
                         {'source_row': 6,
                          'values': {'Capacity': '6.5',
                                     'ShelfID': '7',
                                     'shelf_cleaning_minutes_last_month': '120',
                                     'store_weekly_visitor_count': '1820'}},
                         {'source_row': 7,
                          'values': {'Capacity': '7.5',
                                     'ShelfID': '8',
                                     'shelf_cleaning_minutes_last_month': '60',
                                     'store_weekly_visitor_count': '850'}},
                         {'source_row': 8,
                          'values': {'Capacity': '8.2',
                                     'ShelfID': '9',
                                     'shelf_cleaning_minutes_last_month': '45',
                                     'store_weekly_visitor_count': '1460'}},
                         {'source_row': 9,
                          'values': {'Capacity': '5.7',
                                     'ShelfID': '10',
                                     'shelf_cleaning_minutes_last_month': '120',
                                     'store_weekly_visitor_count': '1820'}}],
             'returned_rows': 10,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['product_catalog_page_views',
                         'ProductName',
                         'Value',
                         'supplier_service_tier',
                         'product_manual_page_count',
                         'Weight'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 20,
             'records': [{'source_row': 0,
                          'values': {'ProductName': 'Smartphone',
                                     'Value': '200',
                                     'Weight': '1.0',
                                     'product_catalog_page_views': '1380',
                                     'product_manual_page_count': '24',
                                     'supplier_service_tier': 'Priority'}},
                         {'source_row': 1,
                          'values': {'ProductName': 'Laptop',
                                     'Value': '1500',
                                     'Weight': '5.0',
                                     'product_catalog_page_views': '180',
                                     'product_manual_page_count': '48',
                                     'supplier_service_tier': 'Priority'}},
                         {'source_row': 2,
                          'values': {'ProductName': 'Headphones',
                                     'Value': '100',
                                     'Weight': '0.5',
                                     'product_catalog_page_views': '340',
                                     'product_manual_page_count': '60',
                                     'supplier_service_tier': 'Premium'}},
                         {'source_row': 3,
                          'values': {'ProductName': 'Camera',
                                     'Value': '800',
                                     'Weight': '2.0',
                                     'product_catalog_page_views': '180',
                                     'product_manual_page_count': '12',
                                     'supplier_service_tier': 'Standard'}},
                         {'source_row': 4,
                          'values': {'ProductName': 'Smartwatch',
                                     'Value': '250',
                                     'Weight': '0.3',
                                     'product_catalog_page_views': '340',
                                     'product_manual_page_count': '48',
                                     'supplier_service_tier': 'Premium'}},
                         {'source_row': 5,
                          'values': {'ProductName': 'Tablet',
                                     'Value': '600',
                                     'Weight': '1.5',
                                     'product_catalog_page_views': '1040',
                                     'product_manual_page_count': '36',
                                     'supplier_service_tier': 'Standard'}},
                         {'source_row': 6,
                          'values': {'ProductName': 'Bluetooth Speaker',
                                     'Value': '150',
                                     'Weight': '1.0',
                                     'product_catalog_page_views': '1040',
                                     'product_manual_page_count': '48',
                                     'supplier_service_tier': 'Priority'}},
                         {'source_row': 7,
                          'values': {'ProductName': 'Keyboard',
                                     'Value': '80',
                                     'Weight': '0.8',
                                     'product_catalog_page_views': '560',
                                     'product_manual_page_count': '36',
                                     'supplier_service_tier': 'Priority'}},
                         {'source_row': 8,
                          'values': {'ProductName': 'Mouse',
                                     'Value': '50',
                                     'Weight': '0.2',
                                     'product_catalog_page_views': '560',
                                     'product_manual_page_count': '12',
                                     'supplier_service_tier': 'Priority'}},
                         {'source_row': 9,
                          'values': {'ProductName': 'Monitor',
                                     'Value': '300',
                                     'Weight': '3.0',
                                     'product_catalog_page_views': '340',
                                     'product_manual_page_count': '60',
                                     'supplier_service_tier': 'Priority'}},
                         {'source_row': 10,
                          'values': {'ProductName': 'Printer',
                                     'Value': '400',
                                     'Weight': '4.0',
                                     'product_catalog_page_views': '340',
                                     'product_manual_page_count': '60',
                                     'supplier_service_tier': 'Priority'}},
                         {'source_row': 11,
                          'values': {'ProductName': 'External Hard Drive',
                                     'Value': '120',
                                     'Weight': '0.5',
                                     'product_catalog_page_views': '560',
                                     'product_manual_page_count': '36',
                                     'supplier_service_tier': 'Priority'}},
                         {'source_row': 12,
                          'values': {'ProductName': 'Router',
                                     'Value': '60',
                                     'Weight': '0.3',
                                     'product_catalog_page_views': '180',
                                     'product_manual_page_count': '60',
                                     'supplier_service_tier': 'Premium'}},
                         {'source_row': 13,
                          'values': {'ProductName': 'Power Bank',
                                     'Value': '40',
                                     'Weight': '0.4',
                                     'product_catalog_page_views': '1380',
                                     'product_manual_page_count': '36',
                                     'supplier_service_tier': 'Priority'}},
                         {'source_row': 14,
                          'values': {'ProductName': 'Memory Card',
                                     'Value': '30',
                                     'Weight': '0.05',
                                     'product_catalog_page_views': '560',
                                     'product_manual_page_count': '12',
                                     'supplier_service_tier': 'Premium'}},
                         {'source_row': 15,
                          'values': {'ProductName': 'USB Flash Drive',
                                     'Value': '25',
                                     'Weight': '0.02',
                                     'product_catalog_page_views': '560',
                                     'product_manual_page_count': '48',
                                     'supplier_service_tier': 'Priority'}},
                         {'source_row': 16,
                          'values': {'ProductName': 'Smart Home Hub',
                                     'Value': '100',
                                     'Weight': '0.6',
                                     'product_catalog_page_views': '180',
                                     'product_manual_page_count': '12',
                                     'supplier_service_tier': 'Premium'}},
                         {'source_row': 17,
                          'values': {'ProductName': 'Gaming Console',
                                     'Value': '500',
                                     'Weight': '4.0',
                                     'product_catalog_page_views': '180',
                                     'product_manual_page_count': '12',
                                     'supplier_service_tier': 'Standard'}},
                         {'source_row': 18,
                          'values': {'ProductName': 'Fitness Tracker',
                                     'Value': '90',
                                     'Weight': '0.2',
                                     'product_catalog_page_views': '790',
                                     'product_manual_page_count': '12',
                                     'supplier_service_tier': 'Priority'}},
                         {'source_row': 19,
                          'values': {'ProductName': 'E-Reader',
                                     'Value': '180',
                                     'Weight': '0.5',
                                     'product_catalog_page_views': '180',
                                     'product_manual_page_count': '60',
                                     'supplier_service_tier': 'Priority'}}],
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

def solve_problem():
    data = CSVQA_DATA
    shelves_table = None
    products_table = None
    for t in data['tables']:
        if t['table_id'] == 'file_0_view_0':
            shelves_table = t
        elif t['table_id'] == 'file_1_view_0':
            products_table = t
    if shelves_table is None or products_table is None:
        raise RuntimeError('Required tables not found in CSVQA_DATA.')
    S = []
    C_s = {}
    for rec in shelves_table['records']:
        shelf_id = rec['values']['ShelfID']
        S.append(shelf_id)
        cap = rec['values']['Capacity']
        try:
            C_s[shelf_id] = float(cap)
        except Exception:
            raise ValueError(f'Invalid capacity for shelf {shelf_id}: {cap}')
    P = []
    v_p = {}
    w_p = {}
    for rec in products_table['records']:
        pname = rec['values']['ProductName']
        P.append(pname)
        val = rec['values']['Value']
        wt = rec['values']['Weight']
        try:
            v_p[pname] = float(val)
            w_p[pname] = float(wt)
        except Exception:
            raise ValueError(f'Invalid value/weight for product {pname}: {val}, {wt}')
    m = gp.Model('Shelf_Product_Allocation')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(S, P, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v_p[p] * x[s, p] for s in S for p in P)), GRB.MAXIMIZE)
    m.addConstrs((gp.quicksum((w_p[p] * x[s, p] for p in P)) <= C_s[s] for s in S), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()