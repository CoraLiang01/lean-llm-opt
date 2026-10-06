CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A supermarket manager needs to select a variety of products to stock in different sections of the store. '
          'Particularly, the store has several sections, each with a display space limit provided in "capacity.csv." '
          'The predefined price and shelf space requirement of each product are detailed in "products.csv." The '
          'objective is to determine the optimal number of units of each product to stock in each section to maximize '
          'the total revenue, while ensuring that the total space used by the products in each section does not exceed '
          'the available capacity. The decision variables x_ij denote the number of units of product j to be placed in '
          'section i.The decision variables must be integers.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'columns': ['SectionID', 'capacity_two_periods_ago', 'previous_period_capacity', 'Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 8,
             'records': [{'source_row': 0,
                          'values': {'Capacity': '100',
                                     'SectionID': '1',
                                     'capacity_two_periods_ago': '101',
                                     'previous_period_capacity': '97'}},
                         {'source_row': 1,
                          'values': {'Capacity': '150',
                                     'SectionID': '2',
                                     'capacity_two_periods_ago': '174',
                                     'previous_period_capacity': '163'}},
                         {'source_row': 2,
                          'values': {'Capacity': '120',
                                     'SectionID': '3',
                                     'capacity_two_periods_ago': '109',
                                     'previous_period_capacity': '142'}},
                         {'source_row': 3,
                          'values': {'Capacity': '130',
                                     'SectionID': '4',
                                     'capacity_two_periods_ago': '129',
                                     'previous_period_capacity': '109'}},
                         {'source_row': 4,
                          'values': {'Capacity': '90',
                                     'SectionID': '5',
                                     'capacity_two_periods_ago': '105',
                                     'previous_period_capacity': '79'}},
                         {'source_row': 5,
                          'values': {'Capacity': '110',
                                     'SectionID': '6',
                                     'capacity_two_periods_ago': '130',
                                     'previous_period_capacity': '131'}},
                         {'source_row': 6,
                          'values': {'Capacity': '160',
                                     'SectionID': '7',
                                     'capacity_two_periods_ago': '182',
                                     'previous_period_capacity': '164'}},
                         {'source_row': 7,
                          'values': {'Capacity': '140',
                                     'SectionID': '8',
                                     'capacity_two_periods_ago': '119',
                                     'previous_period_capacity': '153'}}],
             'returned_rows': 8,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['previous_period_stock_status',
                         'ProductName',
                         'Value',
                         'previous_period_unit_value',
                         'Weight',
                         'previous_period_resource_requirement'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0,
                          'values': {'ProductName': '1',
                                     'Value': '10',
                                     'Weight': '2',
                                     'previous_period_resource_requirement': '3',
                                     'previous_period_stock_status': 'Overstock',
                                     'previous_period_unit_value': '11'}},
                         {'source_row': 1,
                          'values': {'ProductName': '2',
                                     'Value': '15',
                                     'Weight': '3',
                                     'previous_period_resource_requirement': '4',
                                     'previous_period_stock_status': 'Stockout',
                                     'previous_period_unit_value': '12'}},
                         {'source_row': 2,
                          'values': {'ProductName': '3',
                                     'Value': '8',
                                     'Weight': '1',
                                     'previous_period_resource_requirement': '2',
                                     'previous_period_stock_status': 'Stockout',
                                     'previous_period_unit_value': '9'}},
                         {'source_row': 3,
                          'values': {'ProductName': '4',
                                     'Value': '12',
                                     'Weight': '2',
                                     'previous_period_resource_requirement': '3',
                                     'previous_period_stock_status': 'Balanced',
                                     'previous_period_unit_value': '11'}},
                         {'source_row': 4,
                          'values': {'ProductName': '5',
                                     'Value': '20',
                                     'Weight': '4',
                                     'previous_period_resource_requirement': '5',
                                     'previous_period_stock_status': 'Stockout',
                                     'previous_period_unit_value': '16'}},
                         {'source_row': 5,
                          'values': {'ProductName': '6',
                                     'Value': '25',
                                     'Weight': '5',
                                     'previous_period_resource_requirement': '6',
                                     'previous_period_stock_status': 'Stockout',
                                     'previous_period_unit_value': '28'}},
                         {'source_row': 6,
                          'values': {'ProductName': '7',
                                     'Value': '5',
                                     'Weight': '1',
                                     'previous_period_resource_requirement': '2',
                                     'previous_period_stock_status': 'Balanced',
                                     'previous_period_unit_value': '4'}},
                         {'source_row': 7,
                          'values': {'ProductName': '8',
                                     'Value': '30',
                                     'Weight': '6',
                                     'previous_period_resource_requirement': '7',
                                     'previous_period_stock_status': 'Overstock',
                                     'previous_period_unit_value': '27'}},
                         {'source_row': 8,
                          'values': {'ProductName': '9',
                                     'Value': '18',
                                     'Weight': '3',
                                     'previous_period_resource_requirement': '4',
                                     'previous_period_stock_status': 'Overstock',
                                     'previous_period_unit_value': '21'}},
                         {'source_row': 9,
                          'values': {'ProductName': '10',
                                     'Value': '22',
                                     'Weight': '4',
                                     'previous_period_resource_requirement': '5',
                                     'previous_period_stock_status': 'Balanced',
                                     'previous_period_unit_value': '21'}}],
             'returned_rows': 10,
             'role': 'file_1',
             'table_id': 'file_1_view_0'}],
 'validation': {'fallback_reason': "Relationship references an unknown table_id: {'type': 'matrix', 'matrix_table_id': "
                                   "'file_2_view_0', 'row_id_column': 'SectionID', 'row_axis': {'table_id': "
                                   "'file_0_view_0', 'id_column': 'SectionID'}, 'column_axis': {'table_id': "
                                   "'file_1_view_0', 'id_column': 'ProductName'}}",
                'planner_errors': ["Relationship references an unknown table_id: {'type': 'matrix', 'matrix_table_id': "
                                   "'file_2_view_0', 'row_id_column': 'SectionID', 'row_axis': {'table_id': "
                                   "'file_0_view_0', 'id_column': 'SectionID'}, 'column_axis': {'table_id': "
                                   "'file_1_view_0', 'id_column': 'ProductName'}}"],
                'status': 'FALLBACK_FULL_DATA'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    data = CSVQA_DATA
    capacity_table_id = 'file_0_view_0'
    products_table_id = 'file_1_view_0'
    section_records = [rec['values'] for rec in next((t for t in data['tables'] if t['table_id'] == capacity_table_id))['records']]
    product_records = [rec['values'] for rec in next((t for t in data['tables'] if t['table_id'] == products_table_id))['records']]
    S = [rec['SectionID'] for rec in section_records]
    P = [rec['ProductName'] for rec in product_records]
    C_s = {rec['SectionID']: float(rec['Capacity']) for rec in section_records}
    v_p = {rec['ProductName']: float(rec['Value']) for rec in product_records}
    w_p = {rec['ProductName']: float(rec['Weight']) for rec in product_records}
    if set(C_s.keys()) != set(S):
        raise ValueError('Section capacity data missing for some sections.')
    if set(v_p.keys()) != set(P) or set(w_p.keys()) != set(P):
        raise ValueError('Product value or weight data missing for some products.')
    m = gp.Model('Supermarket_Section_Stocking')
    x_keys = [(s, p) for s in S for p in P]
    x = m.addVars(x_keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v_p[p] * x[s, p] for s in S for p in P)), GRB.MAXIMIZE)
    for s in S:
        m.addConstr(gp.quicksum((w_p[p] * x[s, p] for p in P)) <= C_s[s])
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()