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
 'tables': [{'columns': ['archive_revision_number',
                         'Customer',
                         'archive_batch_number',
                         'record_view_count',
                         'demand',
                         'document_page_count'],
             'file_index': 0,
             'file_name': 'demand.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0,
                          'values': {'Customer': 'Customer_1',
                                     'archive_batch_number': '302',
                                     'archive_revision_number': '1',
                                     'demand': '2397',
                                     'document_page_count': '4',
                                     'record_view_count': '58'}},
                         {'source_row': 1,
                          'values': {'Customer': 'Customer_2',
                                     'archive_batch_number': '301',
                                     'archive_revision_number': '2',
                                     'demand': '1889',
                                     'document_page_count': '6',
                                     'record_view_count': '43'}},
                         {'source_row': 2,
                          'values': {'Customer': 'Customer_3',
                                     'archive_batch_number': '301',
                                     'archive_revision_number': '2',
                                     'demand': '2518',
                                     'document_page_count': '6',
                                     'record_view_count': '76'}},
                         {'source_row': 3,
                          'values': {'Customer': 'Customer_4',
                                     'archive_batch_number': '301',
                                     'archive_revision_number': '5',
                                     'demand': '3218',
                                     'document_page_count': '12',
                                     'record_view_count': '12'}},
                         {'source_row': 4,
                          'values': {'Customer': 'Customer_5',
                                     'archive_batch_number': '301',
                                     'archive_revision_number': '5',
                                     'demand': '1813',
                                     'document_page_count': '8',
                                     'record_view_count': '43'}}],
             'returned_rows': 5,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['archive_batch_number',
                         'document_page_count',
                         'archive_revision_number',
                         'Unnamed: 3',
                         'record_view_count',
                         'fixed_costs'],
             'file_index': 1,
             'file_name': 'fixed_cost.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0,
                          'values': {'Unnamed: 3': 'MOUNT AYR',
                                     'archive_batch_number': '304',
                                     'archive_revision_number': '4',
                                     'document_page_count': '4',
                                     'fixed_costs': '96.58',
                                     'record_view_count': '58'}},
                         {'source_row': 1,
                          'values': {'Unnamed: 3': 'WAUKEE',
                                     'archive_batch_number': '302',
                                     'archive_revision_number': '6',
                                     'document_page_count': '2',
                                     'fixed_costs': '94.06',
                                     'record_view_count': '12'}},
                         {'source_row': 2,
                          'values': {'Unnamed: 3': 'WAVERLY',
                                     'archive_batch_number': '301',
                                     'archive_revision_number': '1',
                                     'document_page_count': '4',
                                     'fixed_costs': '94.37',
                                     'record_view_count': '76'}},
                         {'source_row': 3,
                          'values': {'Unnamed: 3': 'PELLA',
                                     'archive_batch_number': '301',
                                     'archive_revision_number': '4',
                                     'document_page_count': '6',
                                     'fixed_costs': '82.88',
                                     'record_view_count': '43'}},
                         {'source_row': 4,
                          'values': {'Unnamed: 3': 'DES MOINES',
                                     'archive_batch_number': '302',
                                     'archive_revision_number': '3',
                                     'document_page_count': '12',
                                     'fixed_costs': '94.95999999999999',
                                     'record_view_count': '76'}}],
             'returned_rows': 5,
             'role': 'file_1',
             'table_id': 'file_1_view_0'},
            {'columns': ['label_print_count',
                         'archive_storage_medium',
                         'record_view_count',
                         'attachment_count',
                         'Unnamed: 4',
                         'CLARINDA',
                         'record_retention_months',
                         'document_page_count',
                         'document_template_family',
                         'record_display_theme',
                         'archive_batch_number',
                         'FORT MADISON',
                         'archive_revision_number',
                         'SIOUX CITY',
                         'audit_reference_number',
                         'TOLEDO',
                         'record_label_font',
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
                                     'Unnamed: 4': 'MOUNT AYR',
                                     'archive_batch_number': '304',
                                     'archive_revision_number': '6',
                                     'archive_storage_medium': 'Hybrid',
                                     'attachment_count': '4',
                                     'audit_reference_number': '1036',
                                     'document_page_count': '2',
                                     'document_template_family': 'Standard',
                                     'label_print_count': '3',
                                     'record_display_theme': 'Slate',
                                     'record_label_font': 'Helvetica',
                                     'record_retention_months': '24',
                                     'record_view_count': '58'}},
                         {'source_row': 1,
                          'values': {'BANCROFT': '90.69',
                                     'CLARINDA': '15.13',
                                     'FORT MADISON': '1.5',
                                     'SIOUX CITY': '1.43',
                                     'TOLEDO': '27.88',
                                     'Unnamed: 4': 'WAUKEE',
                                     'archive_batch_number': '302',
                                     'archive_revision_number': '1',
                                     'archive_storage_medium': 'Paper',
                                     'attachment_count': '5',
                                     'audit_reference_number': '1012',
                                     'document_page_count': '16',
                                     'document_template_family': 'Compact',
                                     'label_print_count': '2',
                                     'record_display_theme': 'Azure',
                                     'record_label_font': 'Arial',
                                     'record_retention_months': '48',
                                     'record_view_count': '58'}},
                         {'source_row': 2,
                          'values': {'BANCROFT': '78.73',
                                     'CLARINDA': '2.34',
                                     'FORT MADISON': '349.34',
                                     'SIOUX CITY': '246.6',
                                     'TOLEDO': '41.3',
                                     'Unnamed: 4': 'WAVERLY',
                                     'archive_batch_number': '301',
                                     'archive_revision_number': '6',
                                     'archive_storage_medium': 'Digital',
                                     'attachment_count': '4',
                                     'audit_reference_number': '1012',
                                     'document_page_count': '8',
                                     'document_template_family': 'Compact',
                                     'label_print_count': '2',
                                     'record_display_theme': 'Slate',
                                     'record_label_font': 'Arial',
                                     'record_retention_months': '48',
                                     'record_view_count': '12'}},
                         {'source_row': 3,
                          'values': {'BANCROFT': '38.93',
                                     'CLARINDA': '1181.6',
                                     'FORT MADISON': '1458.53',
                                     'SIOUX CITY': '1646.36',
                                     'TOLEDO': '1924.55',
                                     'Unnamed: 4': 'PELLA',
                                     'archive_batch_number': '301',
                                     'archive_revision_number': '1',
                                     'archive_storage_medium': 'Paper',
                                     'attachment_count': '2',
                                     'audit_reference_number': '1012',
                                     'document_page_count': '16',
                                     'document_template_family': 'Landscape',
                                     'label_print_count': '6',
                                     'record_display_theme': 'Amber',
                                     'record_label_font': 'Helvetica',
                                     'record_retention_months': '24',
                                     'record_view_count': '76'}},
                         {'source_row': 4,
                          'values': {'BANCROFT': '103.84',
                                     'CLARINDA': '1030.8',
                                     'FORT MADISON': '43.48',
                                     'SIOUX CITY': '932.4299999999999',
                                     'TOLEDO': '55.39',
                                     'Unnamed: 4': 'DES MOINES',
                                     'archive_batch_number': '304',
                                     'archive_revision_number': '4',
                                     'archive_storage_medium': 'Digital',
                                     'attachment_count': '2',
                                     'audit_reference_number': '1024',
                                     'document_page_count': '2',
                                     'document_template_family': 'Compact',
                                     'label_print_count': '1',
                                     'record_display_theme': 'Azure',
                                     'record_label_font': 'Calibri',
                                     'record_retention_months': '24',
                                     'record_view_count': '27'}}],
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
    demand_table = None
    fixed_cost_table = None
    cost_table = None
    for t in data['tables']:
        if t['table_id'] == 'file_0_view_0':
            demand_table = t
        elif t['table_id'] == 'file_1_view_0':
            fixed_cost_table = t
        elif t['table_id'] == 'file_2_view_0':
            cost_table = t
    suppliers = []
    for rec in fixed_cost_table['records']:
        supplier = rec['values']['Unnamed: 3']
        if supplier not in suppliers:
            suppliers.append(supplier)
    stores = []
    for rec in demand_table['records']:
        store = rec['values']['Customer']
        if store not in stores:
            stores.append(store)
    demand = {}
    for rec in demand_table['records']:
        store = rec['values']['Customer']
        d = float(rec['values']['demand'])
        demand[store] = d
    fixed_cost = {}
    for rec in fixed_cost_table['records']:
        supplier = rec['values']['Unnamed: 3']
        f = float(rec['values']['fixed_costs'])
        fixed_cost[supplier] = f
    cost_columns = cost_table['columns']
    store_columns = [col for col in cost_columns if col in stores]
    cost = {}
    for rec in cost_table['records']:
        supplier = rec['values']['Unnamed: 4']
        cost[supplier] = {}
        for store in stores:
            if store in rec['values']:
                cij = float(rec['values'][store])
                cost[supplier][store] = cij
            else:
                raise ValueError(f'Missing cost for supplier {supplier}, store {store}')
    M = sum((demand[j] for j in stores))
    for i in suppliers:
        if i not in fixed_cost:
            raise ValueError(f'Missing fixed cost for supplier {i}')
        if i not in cost:
            raise ValueError(f'Missing cost row for supplier {i}')
        for j in stores:
            if j not in cost[i]:
                raise ValueError(f'Missing cost for supplier {i}, store {j}')
    for j in stores:
        if j not in demand:
            raise ValueError(f'Missing demand for store {j}')
    m = gp.Model('Original_RAG_FLP')
    m.Params.MIPGap = 0.0001
    x_keys = [(i, j) for i in suppliers for j in stores]
    x = m.addVars(x_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    y = m.addVars(suppliers, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in suppliers for j in stores)) + gp.quicksum((fixed_cost[i] * y[i] for i in suppliers)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in suppliers)) == demand[j] for j in stores), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in stores)) <= M * y[i] for i in suppliers), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()