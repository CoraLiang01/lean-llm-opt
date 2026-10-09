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
 'tables': [{'columns': ['inventory_audit_staff_count',
                         'StorageID',
                         'storage_area_cleaning_minutes_last_month',
                         'Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 15,
             'records': [{'source_row': 0,
                          'values': {'Capacity': '1083',
                                     'StorageID': '1',
                                     'inventory_audit_staff_count': '5',
                                     'storage_area_cleaning_minutes_last_month': '120'}},
                         {'source_row': 1,
                          'values': {'Capacity': '1840',
                                     'StorageID': '2',
                                     'inventory_audit_staff_count': '3',
                                     'storage_area_cleaning_minutes_last_month': '300'}},
                         {'source_row': 2,
                          'values': {'Capacity': '770',
                                     'StorageID': '3',
                                     'inventory_audit_staff_count': '2',
                                     'storage_area_cleaning_minutes_last_month': '120'}},
                         {'source_row': 3,
                          'values': {'Capacity': '1299',
                                     'StorageID': '4',
                                     'inventory_audit_staff_count': '5',
                                     'storage_area_cleaning_minutes_last_month': '180'}},
                         {'source_row': 4,
                          'values': {'Capacity': '1259',
                                     'StorageID': '5',
                                     'inventory_audit_staff_count': '5',
                                     'storage_area_cleaning_minutes_last_month': '180'}},
                         {'source_row': 5,
                          'values': {'Capacity': '543',
                                     'StorageID': '6',
                                     'inventory_audit_staff_count': '3',
                                     'storage_area_cleaning_minutes_last_month': '180'}},
                         {'source_row': 6,
                          'values': {'Capacity': '1831',
                                     'StorageID': '7',
                                     'inventory_audit_staff_count': '3',
                                     'storage_area_cleaning_minutes_last_month': '120'}},
                         {'source_row': 7,
                          'values': {'Capacity': '855',
                                     'StorageID': '8',
                                     'inventory_audit_staff_count': '5',
                                     'storage_area_cleaning_minutes_last_month': '360'}},
                         {'source_row': 8,
                          'values': {'Capacity': '619',
                                     'StorageID': '9',
                                     'inventory_audit_staff_count': '5',
                                     'storage_area_cleaning_minutes_last_month': '240'}},
                         {'source_row': 9,
                          'values': {'Capacity': '637',
                                     'StorageID': '10',
                                     'inventory_audit_staff_count': '5',
                                     'storage_area_cleaning_minutes_last_month': '360'}},
                         {'source_row': 10,
                          'values': {'Capacity': '935',
                                     'StorageID': '11',
                                     'inventory_audit_staff_count': '3',
                                     'storage_area_cleaning_minutes_last_month': '240'}},
                         {'source_row': 11,
                          'values': {'Capacity': '626',
                                     'StorageID': '12',
                                     'inventory_audit_staff_count': '4',
                                     'storage_area_cleaning_minutes_last_month': '360'}},
                         {'source_row': 12,
                          'values': {'Capacity': '1457',
                                     'StorageID': '13',
                                     'inventory_audit_staff_count': '6',
                                     'storage_area_cleaning_minutes_last_month': '240'}},
                         {'source_row': 13,
                          'values': {'Capacity': '1198',
                                     'StorageID': '14',
                                     'inventory_audit_staff_count': '2',
                                     'storage_area_cleaning_minutes_last_month': '120'}},
                         {'source_row': 14,
                          'values': {'Capacity': '837',
                                     'StorageID': '15',
                                     'inventory_audit_staff_count': '6',
                                     'storage_area_cleaning_minutes_last_month': '360'}}],
             'returned_rows': 15,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['energy_label_review_count',
                         'ProductName',
                         'supplier_service_tier',
                         'warranty_months',
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
                                     'energy_label_review_count': '2',
                                     'supplier_service_tier': 'Standard',
                                     'warranty_months': '36'}},
                         {'source_row': 1,
                          'values': {'ProductName': 'Portable Unit',
                                     'Value': '1130',
                                     'Weight': '200',
                                     'energy_label_review_count': '3',
                                     'supplier_service_tier': 'Standard',
                                     'warranty_months': '24'}},
                         {'source_row': 2,
                          'values': {'ProductName': 'Split System',
                                     'Value': '1611',
                                     'Weight': '106',
                                     'energy_label_review_count': '1',
                                     'supplier_service_tier': 'Priority',
                                     'warranty_months': '12'}},
                         {'source_row': 3,
                          'values': {'ProductName': 'Ductless System',
                                     'Value': '3368',
                                     'Weight': '256',
                                     'energy_label_review_count': '1',
                                     'supplier_service_tier': 'Standard',
                                     'warranty_months': '36'}},
                         {'source_row': 4,
                          'values': {'ProductName': 'Central AC',
                                     'Value': '2135',
                                     'Weight': '268',
                                     'energy_label_review_count': '2',
                                     'supplier_service_tier': 'Priority',
                                     'warranty_months': '24'}},
                         {'source_row': 5,
                          'values': {'ProductName': 'Hybrid AC',
                                     'Value': '1046',
                                     'Weight': '185',
                                     'energy_label_review_count': '4',
                                     'supplier_service_tier': 'Priority',
                                     'warranty_months': '24'}},
                         {'source_row': 6,
                          'values': {'ProductName': 'Geothermal AC',
                                     'Value': '4030',
                                     'Weight': '299',
                                     'energy_label_review_count': '1',
                                     'supplier_service_tier': 'Priority',
                                     'warranty_months': '48'}},
                         {'source_row': 7,
                          'values': {'ProductName': 'Smart AC',
                                     'Value': '3761',
                                     'Weight': '131',
                                     'energy_label_review_count': '6',
                                     'supplier_service_tier': 'Premium',
                                     'warranty_months': '12'}},
                         {'source_row': 8,
                          'values': {'ProductName': 'Evaporative Cooler',
                                     'Value': '3523',
                                     'Weight': '139',
                                     'energy_label_review_count': '2',
                                     'supplier_service_tier': 'Premium',
                                     'warranty_months': '12'}},
                         {'source_row': 9,
                          'values': {'ProductName': 'Package Unit',
                                     'Value': '1701',
                                     'Weight': '105',
                                     'energy_label_review_count': '3',
                                     'supplier_service_tier': 'Standard',
                                     'warranty_months': '48'}}],
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
    storage_frame = CSVQA_FRAMES['file_0_view_0']
    product_frame = CSVQA_FRAMES['file_1_view_0']
    I = []
    c = {}
    for (_, row) in storage_frame.iterrows():
        storage_id = row['StorageID']
        I.append(storage_id)
        c[storage_id] = float(row['Capacity'])
    J = []
    v = {}
    w = {}
    for (_, row) in product_frame.iterrows():
        product_name = row['ProductName']
        J.append(product_name)
        v[product_name] = float(row['Value'])
        w[product_name] = float(row['Weight'])
    m = gp.Model('Amazon_AC_Storage_Allocation')
    x_vars = m.addVars(I, J, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v[j] * x_vars[i, j] for i in I for j in J)), GRB.MAXIMIZE)
    m.addConstrs((gp.quicksum((w[j] * x_vars[i, j] for j in J)) <= c[i] for i in I), name='')
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