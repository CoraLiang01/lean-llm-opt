CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'In the context of BigMart Sales, the store needs to allocate various types of products into different '
          'display shelves. Specifically, the store has several shelves, each with a capacity limit provided in '
          '‚Äúcapacity.csv.‚Äù The predefined value and weight of each product can be found in ‚Äúproducts.csv.‚Äù The '
          'objective is to determine the optimal number of units of each product to place on each shelf to maximize '
          'the total value of the products across all shelves, while ensuring that the total weight of the products on '
          'each shelf does not exceed its capacity. The decision variables x_ij represent the number of units of '
          'product j to be placed on shelf i.The decision variables must be integers.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'columns': ['previous_period_capacity', 'resource_id', 'resource_capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0,
                          'values': {'previous_period_capacity': '564',
                                     'resource_capacity': '500',
                                     'resource_id': '1'}},
                         {'source_row': 1,
                          'values': {'previous_period_capacity': '679',
                                     'resource_capacity': '700',
                                     'resource_id': '2'}},
                         {'source_row': 2,
                          'values': {'previous_period_capacity': '481',
                                     'resource_capacity': '600',
                                     'resource_id': '3'}},
                         {'source_row': 3,
                          'values': {'previous_period_capacity': '684',
                                     'resource_capacity': '800',
                                     'resource_id': '4'}},
                         {'source_row': 4,
                          'values': {'previous_period_capacity': '623',
                                     'resource_capacity': '550',
                                     'resource_id': '5'}},
                         {'source_row': 5,
                          'values': {'previous_period_capacity': '1019',
                                     'resource_capacity': '900',
                                     'resource_id': '6'}},
                         {'source_row': 6,
                          'values': {'previous_period_capacity': '671',
                                     'resource_capacity': '650',
                                     'resource_id': '7'}},
                         {'source_row': 7,
                          'values': {'previous_period_capacity': '761',
                                     'resource_capacity': '750',
                                     'resource_id': '8'}},
                         {'source_row': 8,
                          'values': {'previous_period_capacity': '951',
                                     'resource_capacity': '820',
                                     'resource_id': '9'}},
                         {'source_row': 9,
                          'values': {'previous_period_capacity': '522',
                                     'resource_capacity': '570',
                                     'resource_id': '10'}}],
             'returned_rows': 10,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['previous_period_stock_status',
                         'item_name',
                         'item_value',
                         'resource_requirement',
                         'previous_period_unit_value'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 20,
             'records': [{'source_row': 0,
                          'values': {'item_name': '1',
                                     'item_value': '50',
                                     'previous_period_stock_status': 'Stockout',
                                     'previous_period_unit_value': '48',
                                     'resource_requirement': '10'}},
                         {'source_row': 1,
                          'values': {'item_name': '2',
                                     'item_value': '70',
                                     'previous_period_stock_status': 'Stockout',
                                     'previous_period_unit_value': '56',
                                     'resource_requirement': '20'}},
                         {'source_row': 2,
                          'values': {'item_name': '3',
                                     'item_value': '30',
                                     'previous_period_stock_status': 'Stockout',
                                     'previous_period_unit_value': '31',
                                     'resource_requirement': '5'}},
                         {'source_row': 3,
                          'values': {'item_name': '4',
                                     'item_value': '60',
                                     'previous_period_stock_status': 'Stockout',
                                     'previous_period_unit_value': '53',
                                     'resource_requirement': '15'}},
                         {'source_row': 4,
                          'values': {'item_name': '5',
                                     'item_value': '80',
                                     'previous_period_stock_status': 'Overstock',
                                     'previous_period_unit_value': '96',
                                     'resource_requirement': '25'}},
                         {'source_row': 5,
                          'values': {'item_name': '6',
                                     'item_value': '90',
                                     'previous_period_stock_status': 'Overstock',
                                     'previous_period_unit_value': '91',
                                     'resource_requirement': '30'}},
                         {'source_row': 6,
                          'values': {'item_name': '7',
                                     'item_value': '40',
                                     'previous_period_stock_status': 'Stockout',
                                     'previous_period_unit_value': '45',
                                     'resource_requirement': '12'}},
                         {'source_row': 7,
                          'values': {'item_name': '8',
                                     'item_value': '100',
                                     'previous_period_stock_status': 'Stockout',
                                     'previous_period_unit_value': '119',
                                     'resource_requirement': '35'}},
                         {'source_row': 8,
                          'values': {'item_name': '9',
                                     'item_value': '55',
                                     'previous_period_stock_status': 'Stockout',
                                     'previous_period_unit_value': '62',
                                     'resource_requirement': '10'}},
                         {'source_row': 9,
                          'values': {'item_name': '10',
                                     'item_value': '75',
                                     'previous_period_stock_status': 'Balanced',
                                     'previous_period_unit_value': '60',
                                     'resource_requirement': '20'}},
                         {'source_row': 10,
                          'values': {'item_name': '11',
                                     'item_value': '65',
                                     'previous_period_stock_status': 'Stockout',
                                     'previous_period_unit_value': '67',
                                     'resource_requirement': '18'}},
                         {'source_row': 11,
                          'values': {'item_name': '12',
                                     'item_value': '95',
                                     'previous_period_stock_status': 'Balanced',
                                     'previous_period_unit_value': '84',
                                     'resource_requirement': '28'}},
                         {'source_row': 12,
                          'values': {'item_name': '13',
                                     'item_value': '45',
                                     'previous_period_stock_status': 'Balanced',
                                     'previous_period_unit_value': '38',
                                     'resource_requirement': '8'}},
                         {'source_row': 13,
                          'values': {'item_name': '14',
                                     'item_value': '85',
                                     'previous_period_stock_status': 'Balanced',
                                     'previous_period_unit_value': '68',
                                     'resource_requirement': '22'}},
                         {'source_row': 14,
                          'values': {'item_name': '15',
                                     'item_value': '70',
                                     'previous_period_stock_status': 'Balanced',
                                     'previous_period_unit_value': '74',
                                     'resource_requirement': '25'}},
                         {'source_row': 15,
                          'values': {'item_name': '16',
                                     'item_value': '110',
                                     'previous_period_stock_status': 'Balanced',
                                     'previous_period_unit_value': '119',
                                     'resource_requirement': '40'}},
                         {'source_row': 16,
                          'values': {'item_name': '17',
                                     'item_value': '50',
                                     'previous_period_stock_status': 'Balanced',
                                     'previous_period_unit_value': '42',
                                     'resource_requirement': '14'}},
                         {'source_row': 17,
                          'values': {'item_name': '18',
                                     'item_value': '60',
                                     'previous_period_stock_status': 'Overstock',
                                     'previous_period_unit_value': '48',
                                     'resource_requirement': '16'}},
                         {'source_row': 18,
                          'values': {'item_name': '19',
                                     'item_value': '120',
                                     'previous_period_stock_status': 'Overstock',
                                     'previous_period_unit_value': '117',
                                     'resource_requirement': '50'}},
                         {'source_row': 19,
                          'values': {'item_name': '20',
                                     'item_value': '100',
                                     'previous_period_stock_status': 'Overstock',
                                     'previous_period_unit_value': '93',
                                     'resource_requirement': '30'}}],
             'returned_rows': 20,
             'role': 'file_1',
             'table_id': 'file_1_view_0'}],
 'validation': {'fallback_reason': "Relationship references an unknown table_id: {'type': 'matrix', 'matrix_table_id': "
                                   "'file_2_view_0', 'row_id_column': 'resource_id', 'row_axis': {'table_id': "
                                   "'file_0_view_0', 'id_column': 'resource_id'}, 'column_axis': {'table_id': "
                                   "'file_1_view_0', 'id_column': 'item_name'}}",
                'planner_errors': ["Relationship references an unknown table_id: {'type': 'matrix', 'matrix_table_id': "
                                   "'file_2_view_0', 'row_id_column': 'resource_id', 'row_axis': {'table_id': "
                                   "'file_0_view_0', 'id_column': 'resource_id'}, 'column_axis': {'table_id': "
                                   "'file_1_view_0', 'id_column': 'item_name'}}"],
                'status': 'FALLBACK_FULL_DATA'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem(CSVQA_FRAMES):
    shelves_frame = CSVQA_FRAMES['file_0_view_0']
    shelves = []
    c_r = {}
    for (source_row, row) in shelves_frame.iterrows():
        shelf_id = row['resource_id']
        shelves.append(shelf_id)
        c_r[shelf_id] = float(row['resource_capacity'])
    products_frame = CSVQA_FRAMES['file_1_view_0']
    products = []
    v_i = {}
    a_i = {}
    for (source_row, row) in products_frame.iterrows():
        product_id = row['item_name']
        products.append(product_id)
        v_i[product_id] = float(row['item_value'])
        a_i[product_id] = float(row['resource_requirement'])
    m = gp.Model('BigMart_Shelf_Allocation')
    m.setParam('MIPGap', 0.0001)
    keys = [(r, i) for r in shelves for i in products]
    x_vars = m.addVars(keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v_i[i] * x_vars[r, i] for r in shelves for i in products)), GRB.MAXIMIZE)
    for r in shelves:
        m.addConstr(gp.quicksum((a_i[i] * x_vars[r, i] for i in products)) <= c_r[r])
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')