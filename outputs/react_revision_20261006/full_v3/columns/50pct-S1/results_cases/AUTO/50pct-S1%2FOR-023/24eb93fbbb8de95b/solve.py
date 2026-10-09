CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The Iowa Department of Commerce requires that any store selling alcohol in bottled form for off-premises '
          'consumption must hold a Class ‚ÄúE‚Äù liquor license, a typical arrangement for most state liquor '
          'regulatory authorities. All alcohol sales from stores registered with the Iowa Department of Commerce are '
          'recorded in the department‚Äôs system, which is publicly released as open data by the State of Iowa. '
          'Several suppliers located in different cities can provide the necessary liquor products to these licensed '
          'stores. Each supplier incurs a fixed cost when starting operations, with the fixed cost data provided in '
          '‚Äúfixed_cost.csv.‚Äù The Department needs to source a unit of each liquor product for the stores from '
          'these suppliers. For each product, the transportation cost per unit from each supplier to each store is '
          'recorded in ‚Äútransportation_costs.csv.‚Äù Additionally, each store has a specific demand for these '
          'products, which is provided in ‚Äúdemand.csv.‚Äù The objective is to determine which suppliers to activate '
          'so that the demand for all liquor products across all licensed stores is met while minimizing the total '
          'cost. The decision variables y_i are binary, indicating whether a supplier is operational (open). The '
          'decision variables x_{ij} represent the quantity of goods that each store S_j sources from supplier F_i.',
 'relationships': [],
 'route': 'FLP',
 'tables': [{'columns': ['archive_revision_number', 'Customer', 'demand'],
             'file_index': 0,
             'file_name': 'demand.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0,
                          'values': {'Customer': 'Customer_1', 'archive_revision_number': '1', 'demand': '2397'}},
                         {'source_row': 1,
                          'values': {'Customer': 'Customer_2', 'archive_revision_number': '2', 'demand': '1889'}},
                         {'source_row': 2,
                          'values': {'Customer': 'Customer_3', 'archive_revision_number': '2', 'demand': '2518'}},
                         {'source_row': 3,
                          'values': {'Customer': 'Customer_4', 'archive_revision_number': '5', 'demand': '3218'}},
                         {'source_row': 4,
                          'values': {'Customer': 'Customer_5', 'archive_revision_number': '5', 'demand': '1813'}}],
             'returned_rows': 5,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['archive_revision_number', 'Unnamed: 1', 'fixed_costs'],
             'file_index': 1,
             'file_name': 'fixed_cost.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0,
                          'values': {'Unnamed: 1': 'MOUNT AYR',
                                     'archive_revision_number': '4',
                                     'fixed_costs': '96.58'}},
                         {'source_row': 1,
                          'values': {'Unnamed: 1': 'WAUKEE', 'archive_revision_number': '6', 'fixed_costs': '94.06'}},
                         {'source_row': 2,
                          'values': {'Unnamed: 1': 'WAVERLY', 'archive_revision_number': '1', 'fixed_costs': '94.37'}},
                         {'source_row': 3,
                          'values': {'Unnamed: 1': 'PELLA', 'archive_revision_number': '4', 'fixed_costs': '82.88'}},
                         {'source_row': 4,
                          'values': {'Unnamed: 1': 'DES MOINES',
                                     'archive_revision_number': '3',
                                     'fixed_costs': '94.95999999999999'}}],
             'returned_rows': 5,
             'role': 'file_1',
             'table_id': 'file_1_view_0'},
            {'columns': ['Unnamed: 0',
                         'CLARINDA',
                         'document_page_count',
                         'record_display_theme',
                         'FORT MADISON',
                         'archive_revision_number',
                         'SIOUX CITY',
                         'TOLEDO',
                         'BANCROFT'],
             'file_index': 2,
             'file_name': 'transportation_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0,
                          'values': {'BANCROFT': '1685.53',
                                     'CLARINDA': '694.6799999999999',
                                     'FORT MADISON': '17.48',
                                     'SIOUX CITY': '20.07',
                                     'TOLEDO': '199.02',
                                     'Unnamed: 0': 'MOUNT AYR',
                                     'archive_revision_number': '6',
                                     'document_page_count': '2',
                                     'record_display_theme': 'Slate'}},
                         {'source_row': 1,
                          'values': {'BANCROFT': '90.69',
                                     'CLARINDA': '15.13',
                                     'FORT MADISON': '1.5',
                                     'SIOUX CITY': '1.43',
                                     'TOLEDO': '27.88',
                                     'Unnamed: 0': 'WAUKEE',
                                     'archive_revision_number': '1',
                                     'document_page_count': '16',
                                     'record_display_theme': 'Azure'}},
                         {'source_row': 2,
                          'values': {'BANCROFT': '78.73',
                                     'CLARINDA': '2.34',
                                     'FORT MADISON': '349.34',
                                     'SIOUX CITY': '246.6',
                                     'TOLEDO': '41.3',
                                     'Unnamed: 0': 'WAVERLY',
                                     'archive_revision_number': '6',
                                     'document_page_count': '8',
                                     'record_display_theme': 'Slate'}},
                         {'source_row': 3,
                          'values': {'BANCROFT': '38.93',
                                     'CLARINDA': '1181.6',
                                     'FORT MADISON': '1458.53',
                                     'SIOUX CITY': '1646.36',
                                     'TOLEDO': '1924.55',
                                     'Unnamed: 0': 'PELLA',
                                     'archive_revision_number': '1',
                                     'document_page_count': '16',
                                     'record_display_theme': 'Amber'}},
                         {'source_row': 4,
                          'values': {'BANCROFT': '103.84',
                                     'CLARINDA': '1030.8',
                                     'FORT MADISON': '43.48',
                                     'SIOUX CITY': '932.4299999999999',
                                     'TOLEDO': '55.39',
                                     'Unnamed: 0': 'DES MOINES',
                                     'archive_revision_number': '4',
                                     'document_page_count': '2',
                                     'record_display_theme': 'Azure'}}],
             'returned_rows': 5,
             'role': 'file_2',
             'table_id': 'file_2_view_0'}],
 'validation': {'fallback_reason': "Matrix validation failed: {'matrix_table_id': 'file_2_view_0', 'shape': [5, 5], "
                                   "'expected_shape': [5, 5], 'row_ids_aligned': True, 'column_ids_aligned': False, "
                                   "'row_mapping_basis': 'exact', 'column_mapping_basis': 'unresolved'}",
                'planner_errors': ["Matrix validation failed: {'matrix_table_id': 'file_2_view_0', 'shape': [5, 5], "
                                   "'expected_shape': [5, 5], 'row_ids_aligned': True, 'column_ids_aligned': False, "
                                   "'row_mapping_basis': 'exact', 'column_mapping_basis': 'unresolved'}"],
                'status': 'FALLBACK_FULL_DATA'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    data = CSVQA_DATA
    demand_table = next((t for t in data['tables'] if t['table_id'] == 'file_0_view_0'))
    fixed_cost_table = next((t for t in data['tables'] if t['table_id'] == 'file_1_view_0'))
    cost_table = next((t for t in data['tables'] if t['table_id'] == 'file_2_view_0'))
    store_columns = ['CLARINDA', 'FORT MADISON', 'SIOUX CITY', 'TOLEDO', 'BANCROFT']
    J = list(store_columns)
    suppliers_from_cost = [rec['values']['Unnamed: 0'] for rec in cost_table['records']]
    suppliers_from_fixed = [rec['values']['Unnamed: 1'] for rec in fixed_cost_table['records']]
    I = []
    seen = set()
    for s in suppliers_from_cost + suppliers_from_fixed:
        if s not in seen:
            I.append(s)
            seen.add(s)
    customer_to_store = dict(zip([rec['values']['Customer'] for rec in demand_table['records']], store_columns))
    d_j = {}
    for rec in demand_table['records']:
        store = customer_to_store[rec['values']['Customer']]
        d_j[store] = float(rec['values']['demand'])
    f_i = {}
    for rec in fixed_cost_table['records']:
        supplier = rec['values']['Unnamed: 1']
        f_i[supplier] = float(rec['values']['fixed_costs'])
    c_ij = {}
    for rec in cost_table['records']:
        supplier = rec['values']['Unnamed: 0']
        c_ij[supplier] = {}
        for store in store_columns:
            c_ij[supplier][store] = float(rec['values'][store])
    M = sum((d_j[j] for j in J))
    for i in I:
        if i not in f_i:
            raise ValueError(f'Missing fixed cost for supplier {i}')
        if i not in c_ij:
            raise ValueError(f'Missing cost row for supplier {i}')
        for j in J:
            if j not in c_ij[i]:
                raise ValueError(f'Missing cost for supplier {i}, store {j}')
    for j in J:
        if j not in d_j:
            raise ValueError(f'Missing demand for store {j}')
    m = gp.Model('Iowa_Liquor_FLP')
    x = m.addVars(I, J, lb=0, vtype=GRB.CONTINUOUS, name='')
    y = m.addVars(I, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((c_ij[i][j] * x[i, j] for i in I for j in J)) + gp.quicksum((f_i[i] * y[i] for i in I)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in I)) == d_j[j] for j in J), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in J)) <= M * y[i] for i in I), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()