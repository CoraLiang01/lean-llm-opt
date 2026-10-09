CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'In the context of retail sales, the store needs to allocate various types of products into different '
          'display shelves. Specifically, the store has several display shelves, each with a capacity limit provided '
          'in “capacity.csv.” The predefined value and weight of each product can be found in “products.csv.” The '
          'objective is to determine the optimal number of units of each product to place on each display shelf to '
          'maximize the total value of the products across all shelves while ensuring that the total weight of the '
          'products on each shelf does not exceed its capacity. The decision variablesx_ijrepresent the number of '
          'units of product j to be placed on shelf i.The decision variables must be integers.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'columns': ['ShelfID', 'Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0, 'values': {'Capacity': '5.0', 'ShelfID': '1'}},
                         {'source_row': 1, 'values': {'Capacity': '7.0', 'ShelfID': '2'}},
                         {'source_row': 2, 'values': {'Capacity': '6.0', 'ShelfID': '3'}},
                         {'source_row': 3, 'values': {'Capacity': '8.0', 'ShelfID': '4'}},
                         {'source_row': 4, 'values': {'Capacity': '5.5', 'ShelfID': '5'}},
                         {'source_row': 5, 'values': {'Capacity': '9.0', 'ShelfID': '6'}},
                         {'source_row': 6, 'values': {'Capacity': '6.5', 'ShelfID': '7'}},
                         {'source_row': 7, 'values': {'Capacity': '7.5', 'ShelfID': '8'}},
                         {'source_row': 8, 'values': {'Capacity': '8.2', 'ShelfID': '9'}},
                         {'source_row': 9, 'values': {'Capacity': '5.7', 'ShelfID': '10'}}],
             'returned_rows': 10,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['ProductName', 'Value', 'Weight'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 20,
             'records': [{'source_row': 0, 'values': {'ProductName': 'Smartphone', 'Value': '200', 'Weight': '1.0'}},
                         {'source_row': 1, 'values': {'ProductName': 'Laptop', 'Value': '1500', 'Weight': '5.0'}},
                         {'source_row': 2, 'values': {'ProductName': 'Headphones', 'Value': '100', 'Weight': '0.5'}},
                         {'source_row': 3, 'values': {'ProductName': 'Camera', 'Value': '800', 'Weight': '2.0'}},
                         {'source_row': 4, 'values': {'ProductName': 'Smartwatch', 'Value': '250', 'Weight': '0.3'}},
                         {'source_row': 5, 'values': {'ProductName': 'Tablet', 'Value': '600', 'Weight': '1.5'}},
                         {'source_row': 6,
                          'values': {'ProductName': 'Bluetooth Speaker', 'Value': '150', 'Weight': '1.0'}},
                         {'source_row': 7, 'values': {'ProductName': 'Keyboard', 'Value': '80', 'Weight': '0.8'}},
                         {'source_row': 8, 'values': {'ProductName': 'Mouse', 'Value': '50', 'Weight': '0.2'}},
                         {'source_row': 9, 'values': {'ProductName': 'Monitor', 'Value': '300', 'Weight': '3.0'}},
                         {'source_row': 10, 'values': {'ProductName': 'Printer', 'Value': '400', 'Weight': '4.0'}},
                         {'source_row': 11,
                          'values': {'ProductName': 'External Hard Drive', 'Value': '120', 'Weight': '0.5'}},
                         {'source_row': 12, 'values': {'ProductName': 'Router', 'Value': '60', 'Weight': '0.3'}},
                         {'source_row': 13, 'values': {'ProductName': 'Power Bank', 'Value': '40', 'Weight': '0.4'}},
                         {'source_row': 14, 'values': {'ProductName': 'Memory Card', 'Value': '30', 'Weight': '0.05'}},
                         {'source_row': 15,
                          'values': {'ProductName': 'USB Flash Drive', 'Value': '25', 'Weight': '0.02'}},
                         {'source_row': 16,
                          'values': {'ProductName': 'Smart Home Hub', 'Value': '100', 'Weight': '0.6'}},
                         {'source_row': 17,
                          'values': {'ProductName': 'Gaming Console', 'Value': '500', 'Weight': '4.0'}},
                         {'source_row': 18,
                          'values': {'ProductName': 'Fitness Tracker', 'Value': '90', 'Weight': '0.2'}},
                         {'source_row': 19, 'values': {'ProductName': 'E-Reader', 'Value': '180', 'Weight': '0.5'}}],
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

def solve_problem():
    shelves_df = CSVQA_FRAMES['file_0_view_0']
    products_df = CSVQA_FRAMES['file_1_view_0']
    S = list(shelves_df['ShelfID'])
    P = list(products_df['ProductName'])
    capacity_s = {}
    for (idx, row) in shelves_df.iterrows():
        shelf_id = row['ShelfID']
        try:
            cap = float(row['Capacity'])
        except Exception:
            raise ValueError(f"Invalid Capacity for ShelfID {shelf_id}: {row['Capacity']}")
        if shelf_id in capacity_s:
            raise ValueError(f'Duplicate ShelfID {shelf_id} in file_0_view_0')
        capacity_s[shelf_id] = cap
    if set(S) != set(capacity_s.keys()):
        raise ValueError('Mismatch in shelf index set and capacity keys')
    value_p = {}
    weight_p = {}
    for (idx, row) in products_df.iterrows():
        prod = row['ProductName']
        try:
            val = float(row['Value'])
            wgt = float(row['Weight'])
        except Exception:
            raise ValueError(f"Invalid Value or Weight for ProductName {prod}: {row['Value']}, {row['Weight']}")
        if prod in value_p or prod in weight_p:
            raise ValueError(f'Duplicate ProductName {prod} in file_1_view_0')
        value_p[prod] = val
        weight_p[prod] = wgt
    if set(P) != set(value_p.keys()) or set(P) != set(weight_p.keys()):
        raise ValueError('Mismatch in product index set and value/weight keys')
    keys = [(s, p) for s in S for p in P]
    m = gp.Model('shelf_product_allocation')
    quantity_vars = m.addVars(keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((value_p[p] * quantity_vars[s, p] for s in S for p in P)), GRB.MAXIMIZE)
    for s in S:
        m.addConstr(gp.quicksum((weight_p[p] * quantity_vars[s, p] for p in P)) <= capacity_s[s], name=f'cap_{s}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for s in S:
            for p in P:
                v = quantity_vars[s, p]
                print(f'{v.VarName} {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()