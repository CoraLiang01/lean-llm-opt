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

def solve_problem(CSVQA_FRAMES):
    df_capacity = CSVQA_FRAMES['file_0_view_0']
    df_products = CSVQA_FRAMES['file_1_view_0']
    I = list(df_capacity['ShelfID'])
    J = list(df_products['ProductName'])
    c_i = {}
    for (idx, row) in df_capacity.iterrows():
        shelf = row['ShelfID']
        try:
            c_i[shelf] = float(row['Capacity'])
        except Exception:
            raise ValueError(f"Non-numeric Capacity for ShelfID {shelf}: {row['Capacity']}")
    v_j = {}
    w_j = {}
    for (idx, row) in df_products.iterrows():
        prod = row['ProductName']
        try:
            v_j[prod] = float(row['Value'])
        except Exception:
            raise ValueError(f"Non-numeric Value for ProductName {prod}: {row['Value']}")
        try:
            w_j[prod] = float(row['Weight'])
        except Exception:
            raise ValueError(f"Non-numeric Weight for ProductName {prod}: {row['Weight']}")
    first_product_row = df_products[df_products['source_row'] == 0]
    if first_product_row.empty:
        raise ValueError('No product with source_row == 0 found in file_1_view_0')
    j_star = first_product_row.iloc[0]['ProductName']
    if set(I) != set(c_i.keys()):
        raise ValueError('Mismatch in ShelfID index set and c_i keys')
    if set(J) != set(v_j.keys()) or set(J) != set(w_j.keys()):
        raise ValueError('Mismatch in ProductName index set and v_j/w_j keys')
    m = gp.Model('retail_display_allocation')
    m.setParam('MIPGap', 0.0001)
    quantity_vars = m.addVars(I, J, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v_j[j] * quantity_vars[i, j] for i in I for j in J)), GRB.MAXIMIZE)
    for i in I:
        m.addConstr(gp.quicksum((w_j[j] * quantity_vars[i, j] for j in J)) <= c_i[i], name=f'cap_{i}')
    m.addConstr(gp.quicksum((quantity_vars[i, j_star] for i in I)) >= 5, name='min_first_product')
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')