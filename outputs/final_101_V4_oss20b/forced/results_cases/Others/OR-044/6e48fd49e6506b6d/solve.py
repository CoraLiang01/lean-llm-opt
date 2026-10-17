CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A supermarket manager needs to select a variety of products to stock in different sections of the store. '
          'Particularly, the store has several sections, each with a display space limit provided in "capacity.csv." '
          'The predefined price and shelf space requirement of each product are detailed in "products.csv." The '
          'objective is to determine the optimal number of units of each product to stock in each section to maximize '
          'the total revenue, while ensuring that the total space used by the products in each section does not exceed '
          'the available capacity. The decision variables x_ij denote the number of units of product j to be placed in '
          'section i.The decision variables must be integers.',
 'relationships': [],
 'route': 'Others',
 'tables': [{'columns': ['SectionID', 'Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 8,
             'records': [{'source_row': 0, 'values': {'Capacity': '100', 'SectionID': '1'}},
                         {'source_row': 1, 'values': {'Capacity': '150', 'SectionID': '2'}},
                         {'source_row': 2, 'values': {'Capacity': '120', 'SectionID': '3'}},
                         {'source_row': 3, 'values': {'Capacity': '130', 'SectionID': '4'}},
                         {'source_row': 4, 'values': {'Capacity': '90', 'SectionID': '5'}},
                         {'source_row': 5, 'values': {'Capacity': '110', 'SectionID': '6'}},
                         {'source_row': 6, 'values': {'Capacity': '160', 'SectionID': '7'}},
                         {'source_row': 7, 'values': {'Capacity': '140', 'SectionID': '8'}}],
             'returned_rows': 8,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['ProductName', 'Value', 'Weight'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0, 'values': {'ProductName': '1', 'Value': '10', 'Weight': '2'}},
                         {'source_row': 1, 'values': {'ProductName': '2', 'Value': '15', 'Weight': '3'}},
                         {'source_row': 2, 'values': {'ProductName': '3', 'Value': '8', 'Weight': '1'}},
                         {'source_row': 3, 'values': {'ProductName': '4', 'Value': '12', 'Weight': '2'}},
                         {'source_row': 4, 'values': {'ProductName': '5', 'Value': '20', 'Weight': '4'}},
                         {'source_row': 5, 'values': {'ProductName': '6', 'Value': '25', 'Weight': '5'}},
                         {'source_row': 6, 'values': {'ProductName': '7', 'Value': '5', 'Weight': '1'}},
                         {'source_row': 7, 'values': {'ProductName': '8', 'Value': '30', 'Weight': '6'}},
                         {'source_row': 8, 'values': {'ProductName': '9', 'Value': '18', 'Weight': '3'}},
                         {'source_row': 9, 'values': {'ProductName': '10', 'Value': '22', 'Weight': '4'}}],
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
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA.get('tables', [])
    capacity_table = None
    products_table = None
    for tbl in tables:
        if tbl.get('table_id') == 'file_0_view_0':
            capacity_table = tbl
        elif tbl.get('table_id') == 'file_1_view_0':
            products_table = tbl
    if capacity_table is None or products_table is None:
        raise ValueError('Required tables not found in CSVQA_DATA')
    capacity = {}
    for rec in capacity_table.get('records', []):
        vals = rec.get('values', {})
        sec_id_str = vals.get('SectionID')
        cap_str = vals.get('Capacity')
        if sec_id_str is None or cap_str is None:
            raise ValueError(f'Missing SectionID or Capacity in record {rec}')
        try:
            sec_id = int(sec_id_str)
            cap = int(cap_str)
        except ValueError:
            raise ValueError(f'Non-integer SectionID or Capacity in record {rec}')
        capacity[sec_id] = cap
    value = {}
    weight = {}
    for rec in products_table.get('records', []):
        vals = rec.get('values', {})
        prod_id_str = vals.get('ProductName')
        val_str = vals.get('Value')
        wt_str = vals.get('Weight')
        if prod_id_str is None or val_str is None or wt_str is None:
            raise ValueError(f'Missing product fields in record {rec}')
        try:
            prod_id = int(prod_id_str)
            val = int(val_str)
            wt = int(wt_str)
        except ValueError:
            raise ValueError(f'Non-integer product fields in record {rec}')
        value[prod_id] = val
        weight[prod_id] = wt
    I = sorted(capacity.keys())
    J = sorted(value.keys())
    for i in I:
        if i not in capacity:
            raise ValueError(f'Capacity missing for section {i}')
    for j in J:
        if j not in value or j not in weight:
            raise ValueError(f'Product data missing for product {j}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    keys = [(i, j) for i in I for j in J]
    x = m.addVars(keys, vtype=GRB.INTEGER, lb=0, name='')
    obj = gurobipy.quicksum((value[j] * x[i, j] for i in I for j in J))
    m.setObjective(obj, GRB.MAXIMIZE)
    for i in I:
        m.addConstr(gurobipy.quicksum((weight[j] * x[i, j] for j in J)) <= capacity[i], name=f'cap_{i}')
    m.optimize()
    return m
m = solve_problem()