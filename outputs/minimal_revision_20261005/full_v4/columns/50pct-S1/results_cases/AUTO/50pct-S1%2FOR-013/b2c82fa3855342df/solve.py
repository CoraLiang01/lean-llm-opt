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
 'tables': [{'columns': ['archive_revision_number', 'StorageID', 'Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 15,
             'records': [{'source_row': 0,
                          'values': {'Capacity': '1083', 'StorageID': '1', 'archive_revision_number': '6'}},
                         {'source_row': 1,
                          'values': {'Capacity': '1840', 'StorageID': '2', 'archive_revision_number': '5'}},
                         {'source_row': 2,
                          'values': {'Capacity': '770', 'StorageID': '3', 'archive_revision_number': '8'}},
                         {'source_row': 3,
                          'values': {'Capacity': '1299', 'StorageID': '4', 'archive_revision_number': '2'}},
                         {'source_row': 4,
                          'values': {'Capacity': '1259', 'StorageID': '5', 'archive_revision_number': '5'}},
                         {'source_row': 5,
                          'values': {'Capacity': '543', 'StorageID': '6', 'archive_revision_number': '4'}},
                         {'source_row': 6,
                          'values': {'Capacity': '1831', 'StorageID': '7', 'archive_revision_number': '1'}},
                         {'source_row': 7,
                          'values': {'Capacity': '855', 'StorageID': '8', 'archive_revision_number': '3'}},
                         {'source_row': 8,
                          'values': {'Capacity': '619', 'StorageID': '9', 'archive_revision_number': '1'}},
                         {'source_row': 9,
                          'values': {'Capacity': '637', 'StorageID': '10', 'archive_revision_number': '6'}},
                         {'source_row': 10,
                          'values': {'Capacity': '935', 'StorageID': '11', 'archive_revision_number': '2'}},
                         {'source_row': 11,
                          'values': {'Capacity': '626', 'StorageID': '12', 'archive_revision_number': '6'}},
                         {'source_row': 12,
                          'values': {'Capacity': '1457', 'StorageID': '13', 'archive_revision_number': '1'}},
                         {'source_row': 13,
                          'values': {'Capacity': '1198', 'StorageID': '14', 'archive_revision_number': '5'}},
                         {'source_row': 14,
                          'values': {'Capacity': '837', 'StorageID': '15', 'archive_revision_number': '8'}}],
             'returned_rows': 15,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['ProductName', 'record_keeper_group', 'archive_revision_number', 'Value', 'Weight'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0,
                          'values': {'ProductName': 'Window Unit',
                                     'Value': '4811',
                                     'Weight': '114',
                                     'archive_revision_number': '5',
                                     'record_keeper_group': 'Team C'}},
                         {'source_row': 1,
                          'values': {'ProductName': 'Portable Unit',
                                     'Value': '1130',
                                     'Weight': '200',
                                     'archive_revision_number': '4',
                                     'record_keeper_group': 'Team A'}},
                         {'source_row': 2,
                          'values': {'ProductName': 'Split System',
                                     'Value': '1611',
                                     'Weight': '106',
                                     'archive_revision_number': '5',
                                     'record_keeper_group': 'Team C'}},
                         {'source_row': 3,
                          'values': {'ProductName': 'Ductless System',
                                     'Value': '3368',
                                     'Weight': '256',
                                     'archive_revision_number': '1',
                                     'record_keeper_group': 'Team A'}},
                         {'source_row': 4,
                          'values': {'ProductName': 'Central AC',
                                     'Value': '2135',
                                     'Weight': '268',
                                     'archive_revision_number': '1',
                                     'record_keeper_group': 'Team A'}},
                         {'source_row': 5,
                          'values': {'ProductName': 'Hybrid AC',
                                     'Value': '1046',
                                     'Weight': '185',
                                     'archive_revision_number': '2',
                                     'record_keeper_group': 'Team C'}},
                         {'source_row': 6,
                          'values': {'ProductName': 'Geothermal AC',
                                     'Value': '4030',
                                     'Weight': '299',
                                     'archive_revision_number': '9',
                                     'record_keeper_group': 'Team C'}},
                         {'source_row': 7,
                          'values': {'ProductName': 'Smart AC',
                                     'Value': '3761',
                                     'Weight': '131',
                                     'archive_revision_number': '3',
                                     'record_keeper_group': 'Team B'}},
                         {'source_row': 8,
                          'values': {'ProductName': 'Evaporative Cooler',
                                     'Value': '3523',
                                     'Weight': '139',
                                     'archive_revision_number': '9',
                                     'record_keeper_group': 'Team B'}},
                         {'source_row': 9,
                          'values': {'ProductName': 'Package Unit',
                                     'Value': '1701',
                                     'Weight': '105',
                                     'archive_revision_number': '2',
                                     'record_keeper_group': 'Team C'}}],
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
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    data = CSVQA_DATA
    capacity_table_id = 'file_0_view_0'
    products_table_id = 'file_1_view_0'
    storage_records = [rec['values'] for rec in next((t for t in data['tables'] if t['table_id'] == capacity_table_id))['records']]
    S = [rec['StorageID'] for rec in storage_records]
    C_s = {rec['StorageID']: int(rec['Capacity']) for rec in storage_records}
    product_records = [rec['values'] for rec in next((t for t in data['tables'] if t['table_id'] == products_table_id))['records']]
    P = [rec['ProductName'] for rec in product_records]
    v_p = {rec['ProductName']: int(rec['Value']) for rec in product_records}
    w_p = {rec['ProductName']: int(rec['Weight']) for rec in product_records}
    for s in S:
        if s not in C_s:
            raise ValueError(f'Missing capacity for storage area {s}')
    for p in P:
        if p not in v_p or p not in w_p:
            raise ValueError(f'Missing value or weight for product {p}')
    m = gp.Model('Amazon_AC_Storage_Allocation')
    keys = [(s, p) for s in S for p in P]
    x = m.addVars(keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v_p[p] * x[s, p] for s in S for p in P)), GRB.MAXIMIZE)
    m.addConstrs((gp.quicksum((w_p[p] * x[s, p] for p in P)) <= C_s[s] for s in S), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')