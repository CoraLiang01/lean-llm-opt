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
 'tables': [{'columns': ['archive_revision_number', 'Customer', 'demand', 'document_page_count'],
             'file_index': 0,
             'file_name': 'demand.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0,
                          'values': {'Customer': 'Customer_1',
                                     'archive_revision_number': '1',
                                     'demand': '2397',
                                     'document_page_count': '4'}},
                         {'source_row': 1,
                          'values': {'Customer': 'Customer_2',
                                     'archive_revision_number': '2',
                                     'demand': '1889',
                                     'document_page_count': '6'}},
                         {'source_row': 2,
                          'values': {'Customer': 'Customer_3',
                                     'archive_revision_number': '2',
                                     'demand': '2518',
                                     'document_page_count': '6'}},
                         {'source_row': 3,
                          'values': {'Customer': 'Customer_4',
                                     'archive_revision_number': '5',
                                     'demand': '3218',
                                     'document_page_count': '12'}},
                         {'source_row': 4,
                          'values': {'Customer': 'Customer_5',
                                     'archive_revision_number': '5',
                                     'demand': '1813',
                                     'document_page_count': '8'}}],
             'returned_rows': 5,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['document_page_count', 'archive_revision_number', 'Unnamed: 2', 'fixed_costs'],
             'file_index': 1,
             'file_name': 'fixed_cost.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0,
                          'values': {'Unnamed: 2': 'MOUNT AYR',
                                     'archive_revision_number': '4',
                                     'document_page_count': '4',
                                     'fixed_costs': '96.58'}},
                         {'source_row': 1,
                          'values': {'Unnamed: 2': 'WAUKEE',
                                     'archive_revision_number': '6',
                                     'document_page_count': '2',
                                     'fixed_costs': '94.06'}},
                         {'source_row': 2,
                          'values': {'Unnamed: 2': 'WAVERLY',
                                     'archive_revision_number': '1',
                                     'document_page_count': '4',
                                     'fixed_costs': '94.37'}},
                         {'source_row': 3,
                          'values': {'Unnamed: 2': 'PELLA',
                                     'archive_revision_number': '4',
                                     'document_page_count': '6',
                                     'fixed_costs': '82.88'}},
                         {'source_row': 4,
                          'values': {'Unnamed: 2': 'DES MOINES',
                                     'archive_revision_number': '3',
                                     'document_page_count': '12',
                                     'fixed_costs': '94.95999999999999'}}],
             'returned_rows': 5,
             'role': 'file_1',
             'table_id': 'file_1_view_0'},
            {'columns': ['archive_storage_medium',
                         'record_view_count',
                         'Unnamed: 2',
                         'CLARINDA',
                         'document_page_count',
                         'record_display_theme',
                         'archive_batch_number',
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
                                     'Unnamed: 2': 'MOUNT AYR',
                                     'archive_batch_number': '304',
                                     'archive_revision_number': '6',
                                     'archive_storage_medium': 'Hybrid',
                                     'document_page_count': '2',
                                     'record_display_theme': 'Slate',
                                     'record_view_count': '58'}},
                         {'source_row': 1,
                          'values': {'BANCROFT': '90.69',
                                     'CLARINDA': '15.13',
                                     'FORT MADISON': '1.5',
                                     'SIOUX CITY': '1.43',
                                     'TOLEDO': '27.88',
                                     'Unnamed: 2': 'WAUKEE',
                                     'archive_batch_number': '302',
                                     'archive_revision_number': '1',
                                     'archive_storage_medium': 'Paper',
                                     'document_page_count': '16',
                                     'record_display_theme': 'Azure',
                                     'record_view_count': '58'}},
                         {'source_row': 2,
                          'values': {'BANCROFT': '78.73',
                                     'CLARINDA': '2.34',
                                     'FORT MADISON': '349.34',
                                     'SIOUX CITY': '246.6',
                                     'TOLEDO': '41.3',
                                     'Unnamed: 2': 'WAVERLY',
                                     'archive_batch_number': '301',
                                     'archive_revision_number': '6',
                                     'archive_storage_medium': 'Digital',
                                     'document_page_count': '8',
                                     'record_display_theme': 'Slate',
                                     'record_view_count': '12'}},
                         {'source_row': 3,
                          'values': {'BANCROFT': '38.93',
                                     'CLARINDA': '1181.6',
                                     'FORT MADISON': '1458.53',
                                     'SIOUX CITY': '1646.36',
                                     'TOLEDO': '1924.55',
                                     'Unnamed: 2': 'PELLA',
                                     'archive_batch_number': '301',
                                     'archive_revision_number': '1',
                                     'archive_storage_medium': 'Paper',
                                     'document_page_count': '16',
                                     'record_display_theme': 'Amber',
                                     'record_view_count': '76'}},
                         {'source_row': 4,
                          'values': {'BANCROFT': '103.84',
                                     'CLARINDA': '1030.8',
                                     'FORT MADISON': '43.48',
                                     'SIOUX CITY': '932.4299999999999',
                                     'TOLEDO': '55.39',
                                     'Unnamed: 2': 'DES MOINES',
                                     'archive_batch_number': '304',
                                     'archive_revision_number': '4',
                                     'archive_storage_medium': 'Digital',
                                     'document_page_count': '2',
                                     'record_display_theme': 'Azure',
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
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB
import pandas as pd

def solve_problem(CSVQA_FRAMES):
    suppliers_1 = [rec['values']['Unnamed: 2'] for rec in CSVQA_FRAMES['file_1_view_0'].records]
    suppliers_2 = [rec['values']['Unnamed: 2'] for rec in CSVQA_FRAMES['file_2_view_0'].records]
    suppliers = []
    seen = set()
    for s in suppliers_1 + suppliers_2:
        if s not in seen:
            suppliers.append(s)
            seen.add(s)
    store_columns = ['BANCROFT', 'CLARINDA', 'FORT MADISON', 'SIOUX CITY', 'TOLEDO']
    stores_0 = [rec['values']['Customer'] for rec in CSVQA_FRAMES['file_0_view_0'].records]
    stores_2 = store_columns
    stores = []
    seen = set()
    for s in stores_0 + stores_2:
        if s not in seen:
            stores.append(s)
            seen.add(s)
    demand = {}
    for rec in CSVQA_FRAMES['file_0_view_0'].records:
        cust = rec['values']['Customer']
        val = rec['values']['demand']
        try:
            demand[cust] = float(val)
        except Exception:
            raise ValueError(f'Invalid demand value for store {cust}: {val}')
    fixed_cost = {}
    for rec in CSVQA_FRAMES['file_1_view_0'].records:
        sup = rec['values']['Unnamed: 2']
        val = rec['values']['fixed_costs']
        try:
            fixed_cost[sup] = float(val)
        except Exception:
            raise ValueError(f'Invalid fixed_costs value for supplier {sup}: {val}')
    cost = {}
    for rec in CSVQA_FRAMES['file_2_view_0'].records:
        sup = rec['values']['Unnamed: 2']
        cost[sup] = {}
        for store in store_columns:
            val = rec['values'][store]
            try:
                cost[sup][store] = float(val)
            except Exception:
                raise ValueError(f'Invalid transportation cost for supplier {sup}, store {store}: {val}')
    M = sum(demand.values())
    for sup in suppliers:
        if sup not in fixed_cost:
            raise ValueError(f'Missing fixed cost for supplier {sup}')
        if sup not in cost:
            raise ValueError(f'Missing transportation cost row for supplier {sup}')
        for store in store_columns:
            if store not in cost[sup]:
                raise ValueError(f'Missing transportation cost for supplier {sup}, store {store}')
    for store in store_columns:
        if store not in stores:
            raise ValueError(f'Store {store} missing from stores list')
    for store in stores_0:
        if store not in demand:
            raise ValueError(f'Demand missing for store {store}')
    m = gp.Model('Iowa_Liquor_FLP')
    m.Params.MIPGap = 0.0001
    quantity_keys = [(i, j) for i in suppliers for j in store_columns]
    quantity_vars = m.addVars(quantity_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    activation_vars = m.addVars(suppliers, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[i][j] * quantity_vars[i, j] for i in suppliers for j in store_columns)) + gp.quicksum((fixed_cost[i] * activation_vars[i] for i in suppliers)), GRB.MINIMIZE)
    for j in store_columns:
        if j not in demand:
            raise ValueError(f'Demand not found for store {j}')
        m.addConstr(gp.quicksum((quantity_vars[i, j] for i in suppliers)) == demand[j], name=f'demand_{j}')
    for i in suppliers:
        m.addConstr(gp.quicksum((quantity_vars[i, j] for j in store_columns)) <= M * activation_vars[i], name=f'activation_{i}')
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')