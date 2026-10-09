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

def solve_problem(CSVQA_FRAMES):
    suppliers = []
    fixed_cost = {}
    file_1 = CSVQA_FRAMES['file_1_view_0']
    for (_, row) in file_1.iterrows():
        supplier = row['Unnamed: 2']
        suppliers.append(supplier)
        try:
            fixed_cost[supplier] = float(row['fixed_costs'])
        except Exception:
            raise ValueError(f"Invalid fixed_costs for supplier {supplier}: {row['fixed_costs']}")
    stores = []
    demand = {}
    D = {}
    file_0 = CSVQA_FRAMES['file_0_view_0']
    for (_, row) in file_0.iterrows():
        store = row['Customer']
        stores.append(store)
        try:
            dval = float(row['demand'])
        except Exception:
            raise ValueError(f"Invalid demand for store {store}: {row['demand']}")
        demand[store] = dval
        D[store] = dval
    file_2 = CSVQA_FRAMES['file_2_view_0']
    cost = {}
    for (_, row) in file_2.iterrows():
        supplier = row['Unnamed: 2']
        cost[supplier] = {}
        for store in stores:
            if store not in row:
                raise ValueError(f'Store {store} not found as a column in transportation_costs.csv')
            try:
                cij = float(row[store])
            except Exception:
                raise ValueError(f'Invalid transportation cost for supplier {supplier}, store {store}: {row[store]}')
            cost[supplier][store] = cij
    if set(suppliers) != set(cost.keys()):
        raise ValueError('Mismatch between suppliers in fixed_cost.csv and transportation_costs.csv')
    for supplier in suppliers:
        if set(stores) != set(cost[supplier].keys()):
            raise ValueError(f'Mismatch between stores in demand.csv and columns in transportation_costs.csv for supplier {supplier}')
    m = gp.Model('Iowa_Liquor_FLP')
    m.setParam('MIPGap', 0.0001)
    x_vars = m.addVars(suppliers, stores, lb=0, vtype=GRB.CONTINUOUS, name='')
    y_vars = m.addVars(suppliers, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[i][j] * x_vars[i, j] for i in suppliers for j in stores)) + gp.quicksum((fixed_cost[i] * y_vars[i] for i in suppliers)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x_vars[i, j] for i in suppliers)) == demand[j] for j in stores), name='')
    m.addConstrs((x_vars[i, j] <= D[j] * y_vars[i] for i in suppliers for j in stores), name='')
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')