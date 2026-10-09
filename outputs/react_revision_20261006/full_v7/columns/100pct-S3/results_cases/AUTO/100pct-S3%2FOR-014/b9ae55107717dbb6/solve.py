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
 'tables': [{'columns': ['capacity_two_periods_ago', 'ShelfID', 'previous_period_capacity', 'Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0,
                          'values': {'Capacity': '5.0',
                                     'ShelfID': '1',
                                     'capacity_two_periods_ago': '5.52600',
                                     'previous_period_capacity': '4.91750'}},
                         {'source_row': 1,
                          'values': {'Capacity': '7.0',
                                     'ShelfID': '2',
                                     'capacity_two_periods_ago': '6.85160',
                                     'previous_period_capacity': '7.69790'}},
                         {'source_row': 2,
                          'values': {'Capacity': '6.0',
                                     'ShelfID': '3',
                                     'capacity_two_periods_ago': '6.1440',
                                     'previous_period_capacity': '5.03040'}},
                         {'source_row': 3,
                          'values': {'Capacity': '8.0',
                                     'ShelfID': '4',
                                     'capacity_two_periods_ago': '9.42160',
                                     'previous_period_capacity': '8.08720'}},
                         {'source_row': 4,
                          'values': {'Capacity': '5.5',
                                     'ShelfID': '5',
                                     'capacity_two_periods_ago': '6.40530',
                                     'previous_period_capacity': '4.67775'}},
                         {'source_row': 5,
                          'values': {'Capacity': '9.0',
                                     'ShelfID': '6',
                                     'capacity_two_periods_ago': '8.54460',
                                     'previous_period_capacity': '9.99450'}},
                         {'source_row': 6,
                          'values': {'Capacity': '6.5',
                                     'ShelfID': '7',
                                     'capacity_two_periods_ago': '7.72330',
                                     'previous_period_capacity': '5.55945'}},
                         {'source_row': 7,
                          'values': {'Capacity': '7.5',
                                     'ShelfID': '8',
                                     'capacity_two_periods_ago': '8.90100',
                                     'previous_period_capacity': '8.8200'}},
                         {'source_row': 8,
                          'values': {'Capacity': '8.2',
                                     'ShelfID': '9',
                                     'capacity_two_periods_ago': '6.68218',
                                     'previous_period_capacity': '9.40294'}},
                         {'source_row': 9,
                          'values': {'Capacity': '5.7',
                                     'ShelfID': '10',
                                     'capacity_two_periods_ago': '5.5347',
                                     'previous_period_capacity': '6.66729'}}],
             'returned_rows': 10,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['previous_period_unit_value',
                         'ProductName',
                         'Value',
                         'previous_period_stock_status',
                         'previous_period_resource_requirement',
                         'Weight'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 20,
             'records': [{'source_row': 0,
                          'values': {'ProductName': 'Smartphone',
                                     'Value': '200',
                                     'Weight': '1.0',
                                     'previous_period_resource_requirement': '0.93290',
                                     'previous_period_stock_status': 'Overstock',
                                     'previous_period_unit_value': '236'}},
                         {'source_row': 1,
                          'values': {'ProductName': 'Laptop',
                                     'Value': '1500',
                                     'Weight': '5.0',
                                     'previous_period_resource_requirement': '5.19600',
                                     'previous_period_stock_status': 'Balanced',
                                     'previous_period_unit_value': '1289'}},
                         {'source_row': 2,
                          'values': {'ProductName': 'Headphones',
                                     'Value': '100',
                                     'Weight': '0.5',
                                     'previous_period_resource_requirement': '0.52775',
                                     'previous_period_stock_status': 'Stockout',
                                     'previous_period_unit_value': '93'}},
                         {'source_row': 3,
                          'values': {'ProductName': 'Camera',
                                     'Value': '800',
                                     'Weight': '2.0',
                                     'previous_period_resource_requirement': '1.80520',
                                     'previous_period_stock_status': 'Overstock',
                                     'previous_period_unit_value': '788'}},
                         {'source_row': 4,
                          'values': {'ProductName': 'Smartwatch',
                                     'Value': '250',
                                     'Weight': '0.3',
                                     'previous_period_resource_requirement': '0.30837',
                                     'previous_period_stock_status': 'Overstock',
                                     'previous_period_unit_value': '279'}},
                         {'source_row': 5,
                          'values': {'ProductName': 'Tablet',
                                     'Value': '600',
                                     'Weight': '1.5',
                                     'previous_period_resource_requirement': '1.44525',
                                     'previous_period_stock_status': 'Overstock',
                                     'previous_period_unit_value': '589'}},
                         {'source_row': 6,
                          'values': {'ProductName': 'Bluetooth Speaker',
                                     'Value': '150',
                                     'Weight': '1.0',
                                     'previous_period_resource_requirement': '1.16120',
                                     'previous_period_stock_status': 'Stockout',
                                     'previous_period_unit_value': '169'}},
                         {'source_row': 7,
                          'values': {'ProductName': 'Keyboard',
                                     'Value': '80',
                                     'Weight': '0.8',
                                     'previous_period_resource_requirement': '0.74656',
                                     'previous_period_stock_status': 'Stockout',
                                     'previous_period_unit_value': '67'}},
                         {'source_row': 8,
                          'values': {'ProductName': 'Mouse',
                                     'Value': '50',
                                     'Weight': '0.2',
                                     'previous_period_resource_requirement': '0.2274',
                                     'previous_period_stock_status': 'Stockout',
                                     'previous_period_unit_value': '55'}},
                         {'source_row': 9,
                          'values': {'ProductName': 'Monitor',
                                     'Value': '300',
                                     'Weight': '3.0',
                                     'previous_period_resource_requirement': '3.17580',
                                     'previous_period_stock_status': 'Overstock',
                                     'previous_period_unit_value': '329'}},
                         {'source_row': 10,
                          'values': {'ProductName': 'Printer',
                                     'Value': '400',
                                     'Weight': '4.0',
                                     'previous_period_resource_requirement': '4.28520',
                                     'previous_period_stock_status': 'Overstock',
                                     'previous_period_unit_value': '462'}},
                         {'source_row': 11,
                          'values': {'ProductName': 'External Hard Drive',
                                     'Value': '120',
                                     'Weight': '0.5',
                                     'previous_period_resource_requirement': '0.59295',
                                     'previous_period_stock_status': 'Balanced',
                                     'previous_period_unit_value': '141'}},
                         {'source_row': 12,
                          'values': {'ProductName': 'Router',
                                     'Value': '60',
                                     'Weight': '0.3',
                                     'previous_period_resource_requirement': '0.31731',
                                     'previous_period_stock_status': 'Balanced',
                                     'previous_period_unit_value': '58'}},
                         {'source_row': 13,
                          'values': {'ProductName': 'Power Bank',
                                     'Value': '40',
                                     'Weight': '0.4',
                                     'previous_period_resource_requirement': '0.36904',
                                     'previous_period_stock_status': 'Stockout',
                                     'previous_period_unit_value': '38'}},
                         {'source_row': 14,
                          'values': {'ProductName': 'Memory Card',
                                     'Value': '30',
                                     'Weight': '0.05',
                                     'previous_period_resource_requirement': '0.051790',
                                     'previous_period_stock_status': 'Overstock',
                                     'previous_period_unit_value': '32'}},
                         {'source_row': 15,
                          'values': {'ProductName': 'USB Flash Drive',
                                     'Value': '25',
                                     'Weight': '0.02',
                                     'previous_period_resource_requirement': '0.02054',
                                     'previous_period_stock_status': 'Stockout',
                                     'previous_period_unit_value': '24'}},
                         {'source_row': 16,
                          'values': {'ProductName': 'Smart Home Hub',
                                     'Value': '100',
                                     'Weight': '0.6',
                                     'previous_period_resource_requirement': '0.58986',
                                     'previous_period_stock_status': 'Stockout',
                                     'previous_period_unit_value': '92'}},
                         {'source_row': 17,
                          'values': {'ProductName': 'Gaming Console',
                                     'Value': '500',
                                     'Weight': '4.0',
                                     'previous_period_resource_requirement': '4.64880',
                                     'previous_period_stock_status': 'Stockout',
                                     'previous_period_unit_value': '510'}},
                         {'source_row': 18,
                          'values': {'ProductName': 'Fitness Tracker',
                                     'Value': '90',
                                     'Weight': '0.2',
                                     'previous_period_resource_requirement': '0.17190',
                                     'previous_period_stock_status': 'Overstock',
                                     'previous_period_unit_value': '80'}},
                         {'source_row': 19,
                          'values': {'ProductName': 'E-Reader',
                                     'Value': '180',
                                     'Weight': '0.5',
                                     'previous_period_resource_requirement': '0.40810',
                                     'previous_period_stock_status': 'Overstock',
                                     'previous_period_unit_value': '165'}}],
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
    import pandas as pd
    shelves_df = CSVQA_FRAMES['file_0_view_0']
    products_df = CSVQA_FRAMES['file_1_view_0']
    S = list(shelves_df['ShelfID'])
    P = list(products_df['ProductName'])
    C_s = {}
    for (idx, row) in shelves_df.iterrows():
        shelf_id = row['ShelfID']
        try:
            C_s[shelf_id] = float(row['Capacity'])
        except Exception:
            raise ValueError(f"Invalid Capacity for ShelfID {shelf_id}: {row['Capacity']}")
    v_p = {}
    w_p = {}
    for (idx, row) in products_df.iterrows():
        product = row['ProductName']
        try:
            v_p[product] = float(row['Value'])
        except Exception:
            raise ValueError(f"Invalid Value for ProductName {product}: {row['Value']}")
        try:
            w_p[product] = float(row['Weight'])
        except Exception:
            raise ValueError(f"Invalid Weight for ProductName {product}: {row['Weight']}")
    if set(S) != set(C_s.keys()):
        raise ValueError('Mismatch in shelves and capacity keys')
    if set(P) != set(v_p.keys()) or set(P) != set(w_p.keys()):
        raise ValueError('Mismatch in products and value/weight keys')
    m = gp.Model('Shelf_Product_Allocation')
    quantity_keys = [(s, p) for s in S for p in P]
    quantity_vars = m.addVars(quantity_keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v_p[p] * quantity_vars[s, p] for s in S for p in P)), GRB.MAXIMIZE)
    m.addConstrs((gp.quicksum((w_p[p] * quantity_vars[s, p] for p in P)) <= C_s[s] for s in S), name='')
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