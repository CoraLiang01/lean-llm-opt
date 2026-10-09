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
 'tables': [{'columns': ['ShelfID', 'previous_period_capacity', 'Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0,
                          'values': {'Capacity': '5.0', 'ShelfID': '1', 'previous_period_capacity': '4.91750'}},
                         {'source_row': 1,
                          'values': {'Capacity': '7.0', 'ShelfID': '2', 'previous_period_capacity': '7.69790'}},
                         {'source_row': 2,
                          'values': {'Capacity': '6.0', 'ShelfID': '3', 'previous_period_capacity': '5.03040'}},
                         {'source_row': 3,
                          'values': {'Capacity': '8.0', 'ShelfID': '4', 'previous_period_capacity': '8.08720'}},
                         {'source_row': 4,
                          'values': {'Capacity': '5.5', 'ShelfID': '5', 'previous_period_capacity': '4.67775'}},
                         {'source_row': 5,
                          'values': {'Capacity': '9.0', 'ShelfID': '6', 'previous_period_capacity': '9.99450'}},
                         {'source_row': 6,
                          'values': {'Capacity': '6.5', 'ShelfID': '7', 'previous_period_capacity': '5.55945'}},
                         {'source_row': 7,
                          'values': {'Capacity': '7.5', 'ShelfID': '8', 'previous_period_capacity': '8.8200'}},
                         {'source_row': 8,
                          'values': {'Capacity': '8.2', 'ShelfID': '9', 'previous_period_capacity': '9.40294'}},
                         {'source_row': 9,
                          'values': {'Capacity': '5.7', 'ShelfID': '10', 'previous_period_capacity': '6.66729'}}],
             'returned_rows': 10,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['previous_period_unit_value',
                         'ProductName',
                         'Value',
                         'previous_period_stock_status',
                         'Weight'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 20,
             'records': [{'source_row': 0,
                          'values': {'ProductName': 'Smartphone',
                                     'Value': '200',
                                     'Weight': '1.0',
                                     'previous_period_stock_status': 'Overstock',
                                     'previous_period_unit_value': '236'}},
                         {'source_row': 1,
                          'values': {'ProductName': 'Laptop',
                                     'Value': '1500',
                                     'Weight': '5.0',
                                     'previous_period_stock_status': 'Balanced',
                                     'previous_period_unit_value': '1289'}},
                         {'source_row': 2,
                          'values': {'ProductName': 'Headphones',
                                     'Value': '100',
                                     'Weight': '0.5',
                                     'previous_period_stock_status': 'Stockout',
                                     'previous_period_unit_value': '93'}},
                         {'source_row': 3,
                          'values': {'ProductName': 'Camera',
                                     'Value': '800',
                                     'Weight': '2.0',
                                     'previous_period_stock_status': 'Overstock',
                                     'previous_period_unit_value': '788'}},
                         {'source_row': 4,
                          'values': {'ProductName': 'Smartwatch',
                                     'Value': '250',
                                     'Weight': '0.3',
                                     'previous_period_stock_status': 'Overstock',
                                     'previous_period_unit_value': '279'}},
                         {'source_row': 5,
                          'values': {'ProductName': 'Tablet',
                                     'Value': '600',
                                     'Weight': '1.5',
                                     'previous_period_stock_status': 'Overstock',
                                     'previous_period_unit_value': '589'}},
                         {'source_row': 6,
                          'values': {'ProductName': 'Bluetooth Speaker',
                                     'Value': '150',
                                     'Weight': '1.0',
                                     'previous_period_stock_status': 'Stockout',
                                     'previous_period_unit_value': '169'}},
                         {'source_row': 7,
                          'values': {'ProductName': 'Keyboard',
                                     'Value': '80',
                                     'Weight': '0.8',
                                     'previous_period_stock_status': 'Stockout',
                                     'previous_period_unit_value': '67'}},
                         {'source_row': 8,
                          'values': {'ProductName': 'Mouse',
                                     'Value': '50',
                                     'Weight': '0.2',
                                     'previous_period_stock_status': 'Stockout',
                                     'previous_period_unit_value': '55'}},
                         {'source_row': 9,
                          'values': {'ProductName': 'Monitor',
                                     'Value': '300',
                                     'Weight': '3.0',
                                     'previous_period_stock_status': 'Overstock',
                                     'previous_period_unit_value': '329'}},
                         {'source_row': 10,
                          'values': {'ProductName': 'Printer',
                                     'Value': '400',
                                     'Weight': '4.0',
                                     'previous_period_stock_status': 'Overstock',
                                     'previous_period_unit_value': '462'}},
                         {'source_row': 11,
                          'values': {'ProductName': 'External Hard Drive',
                                     'Value': '120',
                                     'Weight': '0.5',
                                     'previous_period_stock_status': 'Balanced',
                                     'previous_period_unit_value': '141'}},
                         {'source_row': 12,
                          'values': {'ProductName': 'Router',
                                     'Value': '60',
                                     'Weight': '0.3',
                                     'previous_period_stock_status': 'Balanced',
                                     'previous_period_unit_value': '58'}},
                         {'source_row': 13,
                          'values': {'ProductName': 'Power Bank',
                                     'Value': '40',
                                     'Weight': '0.4',
                                     'previous_period_stock_status': 'Stockout',
                                     'previous_period_unit_value': '38'}},
                         {'source_row': 14,
                          'values': {'ProductName': 'Memory Card',
                                     'Value': '30',
                                     'Weight': '0.05',
                                     'previous_period_stock_status': 'Overstock',
                                     'previous_period_unit_value': '32'}},
                         {'source_row': 15,
                          'values': {'ProductName': 'USB Flash Drive',
                                     'Value': '25',
                                     'Weight': '0.02',
                                     'previous_period_stock_status': 'Stockout',
                                     'previous_period_unit_value': '24'}},
                         {'source_row': 16,
                          'values': {'ProductName': 'Smart Home Hub',
                                     'Value': '100',
                                     'Weight': '0.6',
                                     'previous_period_stock_status': 'Stockout',
                                     'previous_period_unit_value': '92'}},
                         {'source_row': 17,
                          'values': {'ProductName': 'Gaming Console',
                                     'Value': '500',
                                     'Weight': '4.0',
                                     'previous_period_stock_status': 'Stockout',
                                     'previous_period_unit_value': '510'}},
                         {'source_row': 18,
                          'values': {'ProductName': 'Fitness Tracker',
                                     'Value': '90',
                                     'Weight': '0.2',
                                     'previous_period_stock_status': 'Overstock',
                                     'previous_period_unit_value': '80'}},
                         {'source_row': 19,
                          'values': {'ProductName': 'E-Reader',
                                     'Value': '180',
                                     'Weight': '0.5',
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
    I = list(shelves_df['ShelfID'])
    J = list(products_df['ProductName'])
    try:
        c_i = {row['ShelfID']: float(row['Capacity']) for (_, row) in shelves_df.iterrows()}
    except Exception as e:
        raise ValueError(f'Error parsing shelf capacities: {e}')
    try:
        v_j = {row['ProductName']: float(row['Value']) for (_, row) in products_df.iterrows()}
        w_j = {row['ProductName']: float(row['Weight']) for (_, row) in products_df.iterrows()}
    except Exception as e:
        raise ValueError(f'Error parsing product values/weights: {e}')
    if set(c_i.keys()) != set(I):
        raise ValueError('Missing shelf capacities for some ShelfID.')
    if set(v_j.keys()) != set(J) or set(w_j.keys()) != set(J):
        raise ValueError('Missing product value/weight for some ProductName.')
    m = gp.Model('Shelf_Product_Allocation')
    quantity_keys = [(i, j) for i in I for j in J]
    quantity_vars = m.addVars(quantity_keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v_j[j] * quantity_vars[i, j] for i in I for j in J)), GRB.MAXIMIZE)
    m.addConstrs((gp.quicksum((w_j[j] * quantity_vars[i, j] for j in J)) <= c_i[i] for i in I), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')