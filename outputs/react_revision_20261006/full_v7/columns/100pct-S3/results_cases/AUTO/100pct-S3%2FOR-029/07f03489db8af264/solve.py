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
 'tables': [{'columns': ['ShelfID', 'CapacityPreviousLayout', 'Capacity', 'CapacityTwoLayoutsAgo'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0,
                          'values': {'Capacity': '5',
                                     'CapacityPreviousLayout': '5.0165',
                                     'CapacityTwoLayoutsAgo': '5.8745',
                                     'ShelfID': '1'}},
                         {'source_row': 1,
                          'values': {'Capacity': '7',
                                     'CapacityPreviousLayout': '6.4204',
                                     'CapacityTwoLayoutsAgo': '7.0637',
                                     'ShelfID': '2'}},
                         {'source_row': 2,
                          'values': {'Capacity': '6',
                                     'CapacityPreviousLayout': '6.8346',
                                     'CapacityTwoLayoutsAgo': '6.570',
                                     'ShelfID': '3'}},
                         {'source_row': 3,
                          'values': {'Capacity': '8',
                                     'CapacityPreviousLayout': '8.8488',
                                     'CapacityTwoLayoutsAgo': '7.3704',
                                     'ShelfID': '4'}},
                         {'source_row': 4,
                          'values': {'Capacity': '5.5',
                                     'CapacityPreviousLayout': '5.83825',
                                     'CapacityTwoLayoutsAgo': '5.13865',
                                     'ShelfID': '5'}},
                         {'source_row': 5,
                          'values': {'Capacity': '9',
                                     'CapacityPreviousLayout': '10.3554',
                                     'CapacityTwoLayoutsAgo': '8.5203',
                                     'ShelfID': '6'}},
                         {'source_row': 6,
                          'values': {'Capacity': '6.5',
                                     'CapacityPreviousLayout': '7.55430',
                                     'CapacityTwoLayoutsAgo': '5.26045',
                                     'ShelfID': '7'}},
                         {'source_row': 7,
                          'values': {'Capacity': '7.5',
                                     'CapacityPreviousLayout': '6.40350',
                                     'CapacityTwoLayoutsAgo': '8.84700',
                                     'ShelfID': '8'}},
                         {'source_row': 8,
                          'values': {'Capacity': '8.2',
                                     'CapacityPreviousLayout': '8.47142',
                                     'CapacityTwoLayoutsAgo': '6.63790',
                                     'ShelfID': '9'}},
                         {'source_row': 9,
                          'values': {'Capacity': '5.7',
                                     'CapacityPreviousLayout': '6.64506',
                                     'CapacityTwoLayoutsAgo': '6.555',
                                     'ShelfID': '10'}}],
             'returned_rows': 10,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['WeightPreviousPackaging',
                         'ValuePreviousSeason',
                         'CategoryPreviousSeason',
                         'ProductName',
                         'Value',
                         'Weight'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 20,
             'records': [{'source_row': 0,
                          'values': {'CategoryPreviousSeason': 'Accessories',
                                     'ProductName': 'Smartphone',
                                     'Value': '200',
                                     'ValuePreviousSeason': '237',
                                     'Weight': '1',
                                     'WeightPreviousPackaging': '1.132'}},
                         {'source_row': 1,
                          'values': {'CategoryPreviousSeason': 'Electronics',
                                     'ProductName': 'Laptop',
                                     'Value': '1500',
                                     'ValuePreviousSeason': '1476',
                                     'Weight': '5',
                                     'WeightPreviousPackaging': '4.2755'}},
                         {'source_row': 2,
                          'values': {'CategoryPreviousSeason': 'Electronics',
                                     'ProductName': 'Headphones',
                                     'Value': '100',
                                     'ValuePreviousSeason': '93',
                                     'Weight': '0.5',
                                     'WeightPreviousPackaging': '0.56635'}},
                         {'source_row': 3,
                          'values': {'CategoryPreviousSeason': 'Electronics',
                                     'ProductName': 'Camera',
                                     'Value': '800',
                                     'ValuePreviousSeason': '848',
                                     'Weight': '2',
                                     'WeightPreviousPackaging': '2.0106'}},
                         {'source_row': 4,
                          'values': {'CategoryPreviousSeason': 'Electronics',
                                     'ProductName': 'Smartwatch',
                                     'Value': '250',
                                     'ValuePreviousSeason': '256',
                                     'Weight': '0.3',
                                     'WeightPreviousPackaging': '0.31833'}},
                         {'source_row': 5,
                          'values': {'CategoryPreviousSeason': 'Accessories',
                                     'ProductName': 'Tablet',
                                     'Value': '600',
                                     'ValuePreviousSeason': '588',
                                     'Weight': '1.5',
                                     'WeightPreviousPackaging': '1.26570'}},
                         {'source_row': 6,
                          'values': {'CategoryPreviousSeason': 'Accessories',
                                     'ProductName': 'Bluetooth Speaker',
                                     'Value': '150',
                                     'ValuePreviousSeason': '156',
                                     'Weight': '1',
                                     'WeightPreviousPackaging': '1.0724'}},
                         {'source_row': 7,
                          'values': {'CategoryPreviousSeason': 'Office',
                                     'ProductName': 'Keyboard',
                                     'Value': '80',
                                     'ValuePreviousSeason': '81',
                                     'Weight': '0.8',
                                     'WeightPreviousPackaging': '0.64296'}},
                         {'source_row': 8,
                          'values': {'CategoryPreviousSeason': 'Office',
                                     'ProductName': 'Mouse',
                                     'Value': '50',
                                     'ValuePreviousSeason': '46',
                                     'Weight': '0.2',
                                     'WeightPreviousPackaging': '0.16762'}},
                         {'source_row': 9,
                          'values': {'CategoryPreviousSeason': 'Accessories',
                                     'ProductName': 'Monitor',
                                     'Value': '300',
                                     'ValuePreviousSeason': '253',
                                     'Weight': '3',
                                     'WeightPreviousPackaging': '2.4471'}},
                         {'source_row': 10,
                          'values': {'CategoryPreviousSeason': 'Household',
                                     'ProductName': 'Printer',
                                     'Value': '400',
                                     'ValuePreviousSeason': '459',
                                     'Weight': '4',
                                     'WeightPreviousPackaging': '3.9700'}},
                         {'source_row': 11,
                          'values': {'CategoryPreviousSeason': 'Household',
                                     'ProductName': 'External Hard Drive',
                                     'Value': '120',
                                     'ValuePreviousSeason': '106',
                                     'Weight': '0.5',
                                     'WeightPreviousPackaging': '0.51880'}},
                         {'source_row': 12,
                          'values': {'CategoryPreviousSeason': 'Electronics',
                                     'ProductName': 'Router',
                                     'Value': '60',
                                     'ValuePreviousSeason': '55',
                                     'Weight': '0.3',
                                     'WeightPreviousPackaging': '0.26619'}},
                         {'source_row': 13,
                          'values': {'CategoryPreviousSeason': 'Office',
                                     'ProductName': 'Power Bank',
                                     'Value': '40',
                                     'ValuePreviousSeason': '47',
                                     'Weight': '0.4',
                                     'WeightPreviousPackaging': '0.36108'}},
                         {'source_row': 14,
                          'values': {'CategoryPreviousSeason': 'Accessories',
                                     'ProductName': 'Memory Card',
                                     'Value': '30',
                                     'ValuePreviousSeason': '25',
                                     'Weight': '0.05',
                                     'WeightPreviousPackaging': '0.058035'}},
                         {'source_row': 15,
                          'values': {'CategoryPreviousSeason': 'Household',
                                     'ProductName': 'USB Flash Drive',
                                     'Value': '25',
                                     'ValuePreviousSeason': '26',
                                     'Weight': '0.02',
                                     'WeightPreviousPackaging': '0.018236'}},
                         {'source_row': 16,
                          'values': {'CategoryPreviousSeason': 'Household',
                                     'ProductName': 'Smart Home Hub',
                                     'Value': '100',
                                     'ValuePreviousSeason': '116',
                                     'Weight': '0.6',
                                     'WeightPreviousPackaging': '0.53556'}},
                         {'source_row': 17,
                          'values': {'CategoryPreviousSeason': 'Office',
                                     'ProductName': 'Gaming Console',
                                     'Value': '500',
                                     'ValuePreviousSeason': '578',
                                     'Weight': '4',
                                     'WeightPreviousPackaging': '4.7700'}},
                         {'source_row': 18,
                          'values': {'CategoryPreviousSeason': 'Household',
                                     'ProductName': 'Fitness Tracker',
                                     'Value': '90',
                                     'ValuePreviousSeason': '89',
                                     'Weight': '0.2',
                                     'WeightPreviousPackaging': '0.22176'}},
                         {'source_row': 19,
                          'values': {'CategoryPreviousSeason': 'Household',
                                     'ProductName': 'E-Reader',
                                     'Value': '180',
                                     'ValuePreviousSeason': '155',
                                     'Weight': '0.5',
                                     'WeightPreviousPackaging': '0.58055'}}],
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
    df_shelves = CSVQA_FRAMES['file_0_view_0']
    df_products = CSVQA_FRAMES['file_1_view_0']
    shelves = list(df_shelves['ShelfID'])
    products = list(df_products['ProductName'])
    c_i = {}
    for (idx, row) in df_shelves.iterrows():
        shelf_id = row['ShelfID']
        try:
            c_i[shelf_id] = float(row['Capacity'])
        except Exception:
            raise ValueError(f"Invalid Capacity for ShelfID {shelf_id}: {row['Capacity']}")
    v_j = {}
    for (idx, row) in df_products.iterrows():
        prod_id = row['ProductName']
        try:
            v_j[prod_id] = float(row['Value'])
        except Exception:
            raise ValueError(f"Invalid Value for ProductName {prod_id}: {row['Value']}")
    w_j = {}
    for (idx, row) in df_products.iterrows():
        prod_id = row['ProductName']
        try:
            w_j[prod_id] = float(row['Weight'])
        except Exception:
            raise ValueError(f"Invalid Weight for ProductName {prod_id}: {row['Weight']}")
    first_product_row = df_products[df_products['ProductName'] == df_products.iloc[0]['ProductName']]
    if first_product_row.empty:
        raise ValueError('First product not found in products table.')
    j_star = df_products.iloc[0]['ProductName']
    for shelf in shelves:
        if shelf not in c_i:
            raise ValueError(f'Missing capacity for shelf {shelf}')
    for prod in products:
        if prod not in v_j or prod not in w_j:
            raise ValueError(f'Missing value or weight for product {prod}')
    m = gp.Model('retail_display_allocation')
    quantity_keys = [(i, j) for i in shelves for j in products]
    quantity_vars = m.addVars(quantity_keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v_j[j] * quantity_vars[i, j] for i in shelves for j in products)), GRB.MAXIMIZE)
    for i in shelves:
        m.addConstr(gp.quicksum((w_j[j] * quantity_vars[i, j] for j in products)) <= c_i[i])
    m.addConstr(gp.quicksum((quantity_vars[i, j_star] for i in shelves)) >= 5)
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