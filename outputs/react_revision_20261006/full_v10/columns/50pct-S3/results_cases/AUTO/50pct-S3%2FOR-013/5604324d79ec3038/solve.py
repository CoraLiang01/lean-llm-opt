CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'Amazon needs to allocate different types of air conditioners into different warehouse storage areas. '
          'Specifically, Amazon has several storage areas, each with a capacity limit provided in ‚Äúcapacity.csv.‚Äù '
          'The predefined value and size of each air conditioner type can be found in ‚Äúproducts.csv.‚Äù The '
          'objective is to determine the optimal number of units of each air conditioner type to place in each storage '
          'area to maximize the total value of the air conditioners across all areas, while ensuring that the total '
          'size of the units in each area does not exceed its capacity. The decision variablesx_ijrepresent the number '
          'of units of air conditioner type j to be placed in storage area i.The decision variables must be integers.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'columns': ['previous_period_capacity', 'StorageID', 'Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 15,
             'records': [{'source_row': 0,
                          'values': {'Capacity': '1083', 'StorageID': '1', 'previous_period_capacity': '980'}},
                         {'source_row': 1,
                          'values': {'Capacity': '1840', 'StorageID': '2', 'previous_period_capacity': '1920'}},
                         {'source_row': 2,
                          'values': {'Capacity': '770', 'StorageID': '3', 'previous_period_capacity': '870'}},
                         {'source_row': 3,
                          'values': {'Capacity': '1299', 'StorageID': '4', 'previous_period_capacity': '1130'}},
                         {'source_row': 4,
                          'values': {'Capacity': '1259', 'StorageID': '5', 'previous_period_capacity': '1415'}},
                         {'source_row': 5,
                          'values': {'Capacity': '543', 'StorageID': '6', 'previous_period_capacity': '593'}},
                         {'source_row': 6,
                          'values': {'Capacity': '1831', 'StorageID': '7', 'previous_period_capacity': '1930'}},
                         {'source_row': 7,
                          'values': {'Capacity': '855', 'StorageID': '8', 'previous_period_capacity': '860'}},
                         {'source_row': 8,
                          'values': {'Capacity': '619', 'StorageID': '9', 'previous_period_capacity': '710'}},
                         {'source_row': 9,
                          'values': {'Capacity': '637', 'StorageID': '10', 'previous_period_capacity': '560'}},
                         {'source_row': 10,
                          'values': {'Capacity': '935', 'StorageID': '11', 'previous_period_capacity': '891'}},
                         {'source_row': 11,
                          'values': {'Capacity': '626', 'StorageID': '12', 'previous_period_capacity': '683'}},
                         {'source_row': 12,
                          'values': {'Capacity': '1457', 'StorageID': '13', 'previous_period_capacity': '1699'}},
                         {'source_row': 13,
                          'values': {'Capacity': '1198', 'StorageID': '14', 'previous_period_capacity': '1294'}},
                         {'source_row': 14,
                          'values': {'Capacity': '837', 'StorageID': '15', 'previous_period_capacity': '764'}}],
             'returned_rows': 15,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['ProductName',
                         'previous_period_stock_status',
                         'previous_period_unit_value',
                         'Value',
                         'Weight'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0,
                          'values': {'ProductName': 'Window Unit',
                                     'Value': '4811',
                                     'Weight': '114',
                                     'previous_period_stock_status': 'Overstock',
                                     'previous_period_unit_value': '5493'}},
                         {'source_row': 1,
                          'values': {'ProductName': 'Portable Unit',
                                     'Value': '1130',
                                     'Weight': '200',
                                     'previous_period_stock_status': 'Balanced',
                                     'previous_period_unit_value': '1152'}},
                         {'source_row': 2,
                          'values': {'ProductName': 'Split System',
                                     'Value': '1611',
                                     'Weight': '106',
                                     'previous_period_stock_status': 'Overstock',
                                     'previous_period_unit_value': '1471'}},
                         {'source_row': 3,
                          'values': {'ProductName': 'Ductless System',
                                     'Value': '3368',
                                     'Weight': '256',
                                     'previous_period_stock_status': 'Stockout',
                                     'previous_period_unit_value': '3565'}},
                         {'source_row': 4,
                          'values': {'ProductName': 'Central AC',
                                     'Value': '2135',
                                     'Weight': '268',
                                     'previous_period_stock_status': 'Stockout',
                                     'previous_period_unit_value': '2027'}},
                         {'source_row': 5,
                          'values': {'ProductName': 'Hybrid AC',
                                     'Value': '1046',
                                     'Weight': '185',
                                     'previous_period_stock_status': 'Stockout',
                                     'previous_period_unit_value': '1087'}},
                         {'source_row': 6,
                          'values': {'ProductName': 'Geothermal AC',
                                     'Value': '4030',
                                     'Weight': '299',
                                     'previous_period_stock_status': 'Overstock',
                                     'previous_period_unit_value': '3746'}},
                         {'source_row': 7,
                          'values': {'ProductName': 'Smart AC',
                                     'Value': '3761',
                                     'Weight': '131',
                                     'previous_period_stock_status': 'Balanced',
                                     'previous_period_unit_value': '3342'}},
                         {'source_row': 8,
                          'values': {'ProductName': 'Evaporative Cooler',
                                     'Value': '3523',
                                     'Weight': '139',
                                     'previous_period_stock_status': 'Overstock',
                                     'previous_period_unit_value': '3373'}},
                         {'source_row': 9,
                          'values': {'ProductName': 'Package Unit',
                                     'Value': '1701',
                                     'Weight': '105',
                                     'previous_period_stock_status': 'Overstock',
                                     'previous_period_unit_value': '1816'}}],
             'returned_rows': 10,
             'role': 'file_1',
             'table_id': 'file_1_view_0'}],
 'validation': {'fallback_reason': "Relationship references an unknown table_id: {'type': 'matrix', 'matrix_table_id': "
                                   "'file_2_view_0', 'row_id_column': 'StorageID', 'row_axis': {'table_id': "
                                   "'file_0_view_0', 'id_column': 'StorageID'}, 'column_axis': {'table_id': "
                                   "'file_1_view_0', 'id_column': 'ProductName'}}",
                'planner_errors': ["Relationship references an unknown table_id: {'type': 'matrix', 'matrix_table_id': "
                                   "'file_2_view_0', 'row_id_column': 'StorageID', 'row_axis': {'table_id': "
                                   "'file_0_view_0', 'id_column': 'StorageID'}, 'column_axis': {'table_id': "
                                   "'file_1_view_0', 'id_column': 'ProductName'}}"],
                'status': 'FALLBACK_FULL_DATA'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    import pandas as pd
    frame_storage = CSVQA_FRAMES['file_0_view_0']
    frame_products = CSVQA_FRAMES['file_1_view_0']
    I = []
    c = {}
    for (source_row, row) in frame_storage.iterrows():
        storage_id = row['StorageID']
        I.append(storage_id)
        try:
            c[storage_id] = float(row['Capacity'])
        except Exception:
            raise ValueError(f"Invalid Capacity for StorageID {storage_id}: {row['Capacity']}")
    J = []
    v = {}
    w = {}
    for (source_row, row) in frame_products.iterrows():
        product_name = row['ProductName']
        J.append(product_name)
        try:
            v[product_name] = float(row['Value'])
        except Exception:
            raise ValueError(f"Invalid Value for ProductName {product_name}: {row['Value']}")
        try:
            w[product_name] = float(row['Weight'])
        except Exception:
            raise ValueError(f"Invalid Weight for ProductName {product_name}: {row['Weight']}")
    if len(I) == 0 or len(J) == 0:
        raise ValueError('No storage areas or products found.')
    m = gp.Model('Amazon_AC_Storage_Allocation')
    x_keys = [(i, j) for i in I for j in J]
    x_vars = m.addVars(x_keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v[j] * x_vars[i, j] for i in I for j in J)), GRB.MAXIMIZE)
    for i in I:
        m.addConstr(gp.quicksum((w[j] * x_vars[i, j] for j in J)) <= c[i])
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem()