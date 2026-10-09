CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'In BigMart sales, the shop needs to allocate various types of products to different display shelves. The '
          'capacity of each shelf is provided in “capacity.csv,” while each product’s value and weight are in '
          '“products.csv.”The objective is to maximize the total value of products on all shelves without exceeding '
          'any shelf’s capacity. The decision variable  x_{ij} denotes how many units of product 𝑗 are placed on shelf '
          '𝑖.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'columns': ['ShelfID', 'Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0, 'values': {'Capacity': '750', 'ShelfID': '1'}},
                         {'source_row': 1, 'values': {'Capacity': '820', 'ShelfID': '2'}},
                         {'source_row': 2, 'values': {'Capacity': '570', 'ShelfID': '3'}},
                         {'source_row': 3, 'values': {'Capacity': '800', 'ShelfID': '4'}},
                         {'source_row': 4, 'values': {'Capacity': '550', 'ShelfID': '5'}},
                         {'source_row': 5, 'values': {'Capacity': '900', 'ShelfID': '6'}},
                         {'source_row': 6, 'values': {'Capacity': '650', 'ShelfID': '7'}},
                         {'source_row': 7, 'values': {'Capacity': '800', 'ShelfID': '8'}},
                         {'source_row': 8, 'values': {'Capacity': '850', 'ShelfID': '9'}},
                         {'source_row': 9, 'values': {'Capacity': '900', 'ShelfID': '10'}}],
             'returned_rows': 10,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['ProductName', 'Value', 'Weight'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 20,
             'records': [{'source_row': 0, 'values': {'ProductName': '1', 'Value': '55', 'Weight': '10'}},
                         {'source_row': 1, 'values': {'ProductName': '2', 'Value': '75', 'Weight': '20'}},
                         {'source_row': 2, 'values': {'ProductName': '3', 'Value': '65', 'Weight': '5'}},
                         {'source_row': 3, 'values': {'ProductName': '4', 'Value': '60', 'Weight': '15'}},
                         {'source_row': 4, 'values': {'ProductName': '5', 'Value': '80', 'Weight': '25'}},
                         {'source_row': 5, 'values': {'ProductName': '6', 'Value': '90', 'Weight': '35'}},
                         {'source_row': 6, 'values': {'ProductName': '7', 'Value': '40', 'Weight': '45'}},
                         {'source_row': 7, 'values': {'ProductName': '8', 'Value': '100', 'Weight': '55'}},
                         {'source_row': 8, 'values': {'ProductName': '9', 'Value': '55', 'Weight': '65'}},
                         {'source_row': 9, 'values': {'ProductName': '10', 'Value': '75', 'Weight': '20'}},
                         {'source_row': 10, 'values': {'ProductName': '11', 'Value': '110', 'Weight': '18'}},
                         {'source_row': 11, 'values': {'ProductName': '12', 'Value': '50', 'Weight': '28'}},
                         {'source_row': 12, 'values': {'ProductName': '13', 'Value': '60', 'Weight': '8'}},
                         {'source_row': 13, 'values': {'ProductName': '14', 'Value': '120', 'Weight': '28'}},
                         {'source_row': 14, 'values': {'ProductName': '15', 'Value': '70', 'Weight': '25'}},
                         {'source_row': 15, 'values': {'ProductName': '16', 'Value': '110', 'Weight': '40'}},
                         {'source_row': 16, 'values': {'ProductName': '17', 'Value': '50', 'Weight': '55'}},
                         {'source_row': 17, 'values': {'ProductName': '18', 'Value': '60', 'Weight': '70'}},
                         {'source_row': 18, 'values': {'ProductName': '19', 'Value': '120', 'Weight': '85'}},
                         {'source_row': 19, 'values': {'ProductName': '20', 'Value': '100', 'Weight': '100'}}],
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
import pandas as pd

def solve_problem():
    shelves_df = CSVQA_FRAMES['file_0_view_0']
    products_df = CSVQA_FRAMES['file_1_view_0']
    S = list(shelves_df['ShelfID'])
    P = list(products_df['ProductName'])
    C = {}
    for (idx, row) in shelves_df.iterrows():
        shelf_id = row['ShelfID']
        if shelf_id in C:
            raise ValueError(f'Duplicate ShelfID: {shelf_id}')
        try:
            C[shelf_id] = float(row['Capacity'])
        except Exception:
            raise ValueError(f"Invalid Capacity for ShelfID {shelf_id}: {row['Capacity']}")
    V = {}
    W = {}
    for (idx, row) in products_df.iterrows():
        prod_id = row['ProductName']
        if prod_id in V or prod_id in W:
            raise ValueError(f'Duplicate ProductName: {prod_id}')
        try:
            V[prod_id] = float(row['Value'])
        except Exception:
            raise ValueError(f"Invalid Value for ProductName {prod_id}: {row['Value']}")
        try:
            W[prod_id] = float(row['Weight'])
        except Exception:
            raise ValueError(f"Invalid Weight for ProductName {prod_id}: {row['Weight']}")
    if set(S) != set(C.keys()):
        raise ValueError('Mismatch in shelf index set and capacity keys')
    if set(P) != set(V.keys()) or set(P) != set(W.keys()):
        raise ValueError('Mismatch in product index set and value/weight keys')
    m = gp.Model('BigMart_Shelf_Allocation')
    quantity_keys = [(i, j) for i in S for j in P]
    quantity_vars = m.addVars(quantity_keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((V[j] * quantity_vars[i, j] for i in S for j in P)), GRB.MAXIMIZE)
    for i in S:
        m.addConstr(gp.quicksum((W[j] * quantity_vars[i, j] for j in P)) <= C[i], name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem()