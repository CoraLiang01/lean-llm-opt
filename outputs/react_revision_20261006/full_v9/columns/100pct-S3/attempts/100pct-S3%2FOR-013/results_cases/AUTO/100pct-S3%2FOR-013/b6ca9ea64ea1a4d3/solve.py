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
 'tables': [{'columns': ['previous_period_capacity', 'StorageID', 'capacity_two_periods_ago', 'Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 15,
             'records': [{'source_row': 0,
                          'values': {'Capacity': '1083',
                                     'StorageID': '1',
                                     'capacity_two_periods_ago': '1185',
                                     'previous_period_capacity': '980'}},
                         {'source_row': 1,
                          'values': {'Capacity': '1840',
                                     'StorageID': '2',
                                     'capacity_two_periods_ago': '2027',
                                     'previous_period_capacity': '1920'}},
                         {'source_row': 2,
                          'values': {'Capacity': '770',
                                     'StorageID': '3',
                                     'capacity_two_periods_ago': '796',
                                     'previous_period_capacity': '870'}},
                         {'source_row': 3,
                          'values': {'Capacity': '1299',
                                     'StorageID': '4',
                                     'capacity_two_periods_ago': '1205',
                                     'previous_period_capacity': '1130'}},
                         {'source_row': 4,
                          'values': {'Capacity': '1259',
                                     'StorageID': '5',
                                     'capacity_two_periods_ago': '1087',
                                     'previous_period_capacity': '1415'}},
                         {'source_row': 5,
                          'values': {'Capacity': '543',
                                     'StorageID': '6',
                                     'capacity_two_periods_ago': '651',
                                     'previous_period_capacity': '593'}},
                         {'source_row': 6,
                          'values': {'Capacity': '1831',
                                     'StorageID': '7',
                                     'capacity_two_periods_ago': '1975',
                                     'previous_period_capacity': '1930'}},
                         {'source_row': 7,
                          'values': {'Capacity': '855',
                                     'StorageID': '8',
                                     'capacity_two_periods_ago': '873',
                                     'previous_period_capacity': '860'}},
                         {'source_row': 8,
                          'values': {'Capacity': '619',
                                     'StorageID': '9',
                                     'capacity_two_periods_ago': '626',
                                     'previous_period_capacity': '710'}},
                         {'source_row': 9,
                          'values': {'Capacity': '637',
                                     'StorageID': '10',
                                     'capacity_two_periods_ago': '654',
                                     'previous_period_capacity': '560'}},
                         {'source_row': 10,
                          'values': {'Capacity': '935',
                                     'StorageID': '11',
                                     'capacity_two_periods_ago': '884',
                                     'previous_period_capacity': '891'}},
                         {'source_row': 11,
                          'values': {'Capacity': '626',
                                     'StorageID': '12',
                                     'capacity_two_periods_ago': '515',
                                     'previous_period_capacity': '683'}},
                         {'source_row': 12,
                          'values': {'Capacity': '1457',
                                     'StorageID': '13',
                                     'capacity_two_periods_ago': '1729',
                                     'previous_period_capacity': '1699'}},
                         {'source_row': 13,
                          'values': {'Capacity': '1198',
                                     'StorageID': '14',
                                     'capacity_two_periods_ago': '994',
                                     'previous_period_capacity': '1294'}},
                         {'source_row': 14,
                          'values': {'Capacity': '837',
                                     'StorageID': '15',
                                     'capacity_two_periods_ago': '919',
                                     'previous_period_capacity': '764'}}],
             'returned_rows': 15,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['previous_period_resource_requirement',
                         'ProductName',
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
                                     'previous_period_resource_requirement': '111',
                                     'previous_period_stock_status': 'Overstock',
                                     'previous_period_unit_value': '5493'}},
                         {'source_row': 1,
                          'values': {'ProductName': 'Portable Unit',
                                     'Value': '1130',
                                     'Weight': '200',
                                     'previous_period_resource_requirement': '231',
                                     'previous_period_stock_status': 'Balanced',
                                     'previous_period_unit_value': '1152'}},
                         {'source_row': 2,
                          'values': {'ProductName': 'Split System',
                                     'Value': '1611',
                                     'Weight': '106',
                                     'previous_period_resource_requirement': '118',
                                     'previous_period_stock_status': 'Overstock',
                                     'previous_period_unit_value': '1471'}},
                         {'source_row': 3,
                          'values': {'ProductName': 'Ductless System',
                                     'Value': '3368',
                                     'Weight': '256',
                                     'previous_period_resource_requirement': '303',
                                     'previous_period_stock_status': 'Stockout',
                                     'previous_period_unit_value': '3565'}},
                         {'source_row': 4,
                          'values': {'ProductName': 'Central AC',
                                     'Value': '2135',
                                     'Weight': '268',
                                     'previous_period_resource_requirement': '292',
                                     'previous_period_stock_status': 'Stockout',
                                     'previous_period_unit_value': '2027'}},
                         {'source_row': 5,
                          'values': {'ProductName': 'Hybrid AC',
                                     'Value': '1046',
                                     'Weight': '185',
                                     'previous_period_resource_requirement': '181',
                                     'previous_period_stock_status': 'Stockout',
                                     'previous_period_unit_value': '1087'}},
                         {'source_row': 6,
                          'values': {'ProductName': 'Geothermal AC',
                                     'Value': '4030',
                                     'Weight': '299',
                                     'previous_period_resource_requirement': '318',
                                     'previous_period_stock_status': 'Overstock',
                                     'previous_period_unit_value': '3746'}},
                         {'source_row': 7,
                          'values': {'ProductName': 'Smart AC',
                                     'Value': '3761',
                                     'Weight': '131',
                                     'previous_period_resource_requirement': '112',
                                     'previous_period_stock_status': 'Balanced',
                                     'previous_period_unit_value': '3342'}},
                         {'source_row': 8,
                          'values': {'ProductName': 'Evaporative Cooler',
                                     'Value': '3523',
                                     'Weight': '139',
                                     'previous_period_resource_requirement': '163',
                                     'previous_period_stock_status': 'Overstock',
                                     'previous_period_unit_value': '3373'}},
                         {'source_row': 9,
                          'values': {'ProductName': 'Package Unit',
                                     'Value': '1701',
                                     'Weight': '105',
                                     'previous_period_resource_requirement': '113',
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

def solve_problem(CSVQA_FRAMES):
    cap_frame = CSVQA_FRAMES['file_0_view_0']
    I = []
    c = {}
    for (_, row) in cap_frame.iterrows():
        storage_id = row['StorageID']
        I.append(storage_id)
        c[storage_id] = float(row['Capacity'])
    prod_frame = CSVQA_FRAMES['file_1_view_0']
    J = []
    v = {}
    w = {}
    for (_, row) in prod_frame.iterrows():
        prod_name = row['ProductName']
        J.append(prod_name)
        v[prod_name] = float(row['Value'])
        w[prod_name] = float(row['Weight'])
    if len(I) == 0 or len(J) == 0:
        raise ValueError('No storage areas or product types found in input data.')
    for i in I:
        if i not in c:
            raise ValueError(f'Missing capacity for storage area {i}.')
    for j in J:
        if j not in v or j not in w:
            raise ValueError(f'Missing value or weight for product {j}.')
    m = gp.Model('Amazon_AC_Storage')
    m.Params.MIPGap = 0.0001
    x_vars = m.addVars(I, J, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v[j] * x_vars[i, j] for i in I for j in J)), GRB.MAXIMIZE)
    m.addConstrs((gp.quicksum((w[j] * x_vars[i, j] for j in J)) <= c[i] for i in I), name='')
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')