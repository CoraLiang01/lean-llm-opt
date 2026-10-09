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
 'tables': [{'columns': ['archive_revision_number', 'resource_id', 'resource_capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0,
                          'values': {'archive_revision_number': '5', 'resource_capacity': '500', 'resource_id': '1'}},
                         {'source_row': 1,
                          'values': {'archive_revision_number': '8', 'resource_capacity': '700', 'resource_id': '2'}},
                         {'source_row': 2,
                          'values': {'archive_revision_number': '1', 'resource_capacity': '600', 'resource_id': '3'}},
                         {'source_row': 3,
                          'values': {'archive_revision_number': '2', 'resource_capacity': '800', 'resource_id': '4'}},
                         {'source_row': 4,
                          'values': {'archive_revision_number': '2', 'resource_capacity': '550', 'resource_id': '5'}},
                         {'source_row': 5,
                          'values': {'archive_revision_number': '7', 'resource_capacity': '900', 'resource_id': '6'}},
                         {'source_row': 6,
                          'values': {'archive_revision_number': '4', 'resource_capacity': '650', 'resource_id': '7'}},
                         {'source_row': 7,
                          'values': {'archive_revision_number': '1', 'resource_capacity': '750', 'resource_id': '8'}},
                         {'source_row': 8,
                          'values': {'archive_revision_number': '2', 'resource_capacity': '820', 'resource_id': '9'}},
                         {'source_row': 9,
                          'values': {'archive_revision_number': '2', 'resource_capacity': '570', 'resource_id': '10'}}],
             'returned_rows': 10,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['record_keeper_group',
                         'item_name',
                         'item_value',
                         'resource_requirement',
                         'archive_revision_number'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 20,
             'records': [{'source_row': 0,
                          'values': {'archive_revision_number': '4',
                                     'item_name': '1',
                                     'item_value': '50',
                                     'record_keeper_group': 'Team B',
                                     'resource_requirement': '10'}},
                         {'source_row': 1,
                          'values': {'archive_revision_number': '3',
                                     'item_name': '2',
                                     'item_value': '70',
                                     'record_keeper_group': 'Team A',
                                     'resource_requirement': '20'}},
                         {'source_row': 2,
                          'values': {'archive_revision_number': '8',
                                     'item_name': '3',
                                     'item_value': '30',
                                     'record_keeper_group': 'Team C',
                                     'resource_requirement': '5'}},
                         {'source_row': 3,
                          'values': {'archive_revision_number': '5',
                                     'item_name': '4',
                                     'item_value': '60',
                                     'record_keeper_group': 'Team B',
                                     'resource_requirement': '15'}},
                         {'source_row': 4,
                          'values': {'archive_revision_number': '2',
                                     'item_name': '5',
                                     'item_value': '80',
                                     'record_keeper_group': 'Team B',
                                     'resource_requirement': '25'}},
                         {'source_row': 5,
                          'values': {'archive_revision_number': '1',
                                     'item_name': '6',
                                     'item_value': '90',
                                     'record_keeper_group': 'Team C',
                                     'resource_requirement': '30'}},
                         {'source_row': 6,
                          'values': {'archive_revision_number': '5',
                                     'item_name': '7',
                                     'item_value': '40',
                                     'record_keeper_group': 'Team B',
                                     'resource_requirement': '12'}},
                         {'source_row': 7,
                          'values': {'archive_revision_number': '8',
                                     'item_name': '8',
                                     'item_value': '100',
                                     'record_keeper_group': 'Team B',
                                     'resource_requirement': '35'}},
                         {'source_row': 8,
                          'values': {'archive_revision_number': '2',
                                     'item_name': '9',
                                     'item_value': '55',
                                     'record_keeper_group': 'Team C',
                                     'resource_requirement': '10'}},
                         {'source_row': 9,
                          'values': {'archive_revision_number': '6',
                                     'item_name': '10',
                                     'item_value': '75',
                                     'record_keeper_group': 'Team C',
                                     'resource_requirement': '20'}},
                         {'source_row': 10,
                          'values': {'archive_revision_number': '8',
                                     'item_name': '11',
                                     'item_value': '65',
                                     'record_keeper_group': 'Team C',
                                     'resource_requirement': '18'}},
                         {'source_row': 11,
                          'values': {'archive_revision_number': '3',
                                     'item_name': '12',
                                     'item_value': '95',
                                     'record_keeper_group': 'Team B',
                                     'resource_requirement': '28'}},
                         {'source_row': 12,
                          'values': {'archive_revision_number': '8',
                                     'item_name': '13',
                                     'item_value': '45',
                                     'record_keeper_group': 'Team A',
                                     'resource_requirement': '8'}},
                         {'source_row': 13,
                          'values': {'archive_revision_number': '4',
                                     'item_name': '14',
                                     'item_value': '85',
                                     'record_keeper_group': 'Team A',
                                     'resource_requirement': '22'}},
                         {'source_row': 14,
                          'values': {'archive_revision_number': '3',
                                     'item_name': '15',
                                     'item_value': '70',
                                     'record_keeper_group': 'Team B',
                                     'resource_requirement': '25'}},
                         {'source_row': 15,
                          'values': {'archive_revision_number': '3',
                                     'item_name': '16',
                                     'item_value': '110',
                                     'record_keeper_group': 'Team A',
                                     'resource_requirement': '40'}},
                         {'source_row': 16,
                          'values': {'archive_revision_number': '2',
                                     'item_name': '17',
                                     'item_value': '50',
                                     'record_keeper_group': 'Team A',
                                     'resource_requirement': '14'}},
                         {'source_row': 17,
                          'values': {'archive_revision_number': '5',
                                     'item_name': '18',
                                     'item_value': '60',
                                     'record_keeper_group': 'Team B',
                                     'resource_requirement': '16'}},
                         {'source_row': 18,
                          'values': {'archive_revision_number': '8',
                                     'item_name': '19',
                                     'item_value': '120',
                                     'record_keeper_group': 'Team C',
                                     'resource_requirement': '50'}},
                         {'source_row': 19,
                          'values': {'archive_revision_number': '8',
                                     'item_name': '20',
                                     'item_value': '100',
                                     'record_keeper_group': 'Team B',
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
    C_s = {}
    for (_, row) in shelves_frame.iterrows():
        shelf_id = row['resource_id']
        shelves.append(shelf_id)
        C_s[shelf_id] = float(row['resource_capacity'])
    products_frame = CSVQA_FRAMES['file_1_view_0']
    products = []
    v_p = {}
    w_p = {}
    for (_, row) in products_frame.iterrows():
        product_id = row['item_name']
        products.append(product_id)
        v_p[product_id] = float(row['item_value'])
        w_p[product_id] = float(row['resource_requirement'])
    if len(shelves) == 0 or len(products) == 0:
        raise ValueError('No shelves or products found in the input data.')
    for s in shelves:
        if s not in C_s:
            raise ValueError(f'Missing capacity for shelf {s}')
    for p in products:
        if p not in v_p or p not in w_p:
            raise ValueError(f'Missing value or weight for product {p}')
    m = gp.Model('BigMart_Shelf_Allocation')
    quantity_vars = m.addVars(shelves, products, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v_p[p] * quantity_vars[s, p] for s in shelves for p in products)), GRB.MAXIMIZE)
    m.addConstrs((gp.quicksum((w_p[p] * quantity_vars[s, p] for p in products)) <= C_s[s] for s in shelves), name='')
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