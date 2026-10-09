CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A retail chain, ‚ÄúGreenMart,‚Äù operates several warehouses that supply products to its various store '
          'locations. The daily demand for each store is provided in ‚Äúcustomer_demand.csv,‚Äù while the daily supply '
          'capacity of each warehouse is detailed in ‚Äúsupply_capacity.csv.‚Äù The cost of transporting each unit of '
          'product from each warehouse to each store is recorded in ‚Äútransportation_costs.csv.‚Äù The objective is '
          'to determine the optimal quantity of products to be shipped from each warehouse to each GreenMart store, '
          'ensuring that all store demands are met without exceeding the supply capacity of any warehouse, while '
          'minimizing the total transportation cost.',
 'relationships': [],
 'route': 'TP',
 'tables': [{'columns': ['customer_id',
                         'document_page_count',
                         'archive_revision_number',
                         'record_view_count',
                         'demand_units',
                         'archive_batch_number'],
             'file_index': 0,
             'file_name': 'customer_demand.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0,
                          'values': {'archive_batch_number': '301',
                                     'archive_revision_number': '6',
                                     'customer_id': 'D1',
                                     'demand_units': '428',
                                     'document_page_count': '2',
                                     'record_view_count': '43'}},
                         {'source_row': 1,
                          'values': {'archive_batch_number': '302',
                                     'archive_revision_number': '6',
                                     'customer_id': 'D2',
                                     'demand_units': '217',
                                     'document_page_count': '12',
                                     'record_view_count': '43'}},
                         {'source_row': 2,
                          'values': {'archive_batch_number': '304',
                                     'archive_revision_number': '3',
                                     'customer_id': 'D3',
                                     'demand_units': '214',
                                     'document_page_count': '6',
                                     'record_view_count': '76'}},
                         {'source_row': 3,
                          'values': {'archive_batch_number': '305',
                                     'archive_revision_number': '4',
                                     'customer_id': 'D4',
                                     'demand_units': '380',
                                     'document_page_count': '6',
                                     'record_view_count': '12'}},
                         {'source_row': 4,
                          'values': {'archive_batch_number': '305',
                                     'archive_revision_number': '1',
                                     'customer_id': 'D5',
                                     'demand_units': '254',
                                     'document_page_count': '8',
                                     'record_view_count': '58'}}],
             'returned_rows': 5,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['record_view_count',
                         'archive_revision_number',
                         'supplier_id',
                         'document_page_count',
                         'archive_batch_number',
                         'supply_capacity_units'],
             'file_index': 1,
             'file_name': 'supply_capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0,
                          'values': {'archive_batch_number': '304',
                                     'archive_revision_number': '6',
                                     'document_page_count': '8',
                                     'record_view_count': '58',
                                     'supplier_id': 'S1',
                                     'supply_capacity_units': '428'}},
                         {'source_row': 1,
                          'values': {'archive_batch_number': '304',
                                     'archive_revision_number': '2',
                                     'document_page_count': '6',
                                     'record_view_count': '76',
                                     'supplier_id': 'S2',
                                     'supply_capacity_units': '217'}},
                         {'source_row': 2,
                          'values': {'archive_batch_number': '303',
                                     'archive_revision_number': '1',
                                     'document_page_count': '2',
                                     'record_view_count': '76',
                                     'supplier_id': 'S3',
                                     'supply_capacity_units': '214'}},
                         {'source_row': 3,
                          'values': {'archive_batch_number': '302',
                                     'archive_revision_number': '6',
                                     'document_page_count': '8',
                                     'record_view_count': '58',
                                     'supplier_id': 'S4',
                                     'supply_capacity_units': '380'}},
                         {'source_row': 4,
                          'values': {'archive_batch_number': '302',
                                     'archive_revision_number': '2',
                                     'document_page_count': '4',
                                     'record_view_count': '58',
                                     'supplier_id': 'S5',
                                     'supply_capacity_units': '254'}}],
             'returned_rows': 5,
             'role': 'file_1',
             'table_id': 'file_1_view_0'},
            {'columns': ['record_label_font',
                         'supplier_id',
                         'document_page_count',
                         'document_template_family',
                         'transportation_cost_to_D1',
                         'transportation_cost_to_D2',
                         'label_print_count',
                         'archive_batch_number',
                         'record_retention_months',
                         'archive_storage_medium',
                         'transportation_cost_to_D3',
                         'attachment_count',
                         'record_display_theme',
                         'transportation_cost_to_D4',
                         'audit_reference_number',
                         'transportation_cost_to_D5',
                         'record_view_count',
                         'archive_revision_number'],
             'file_index': 2,
             'file_name': 'transportation_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0,
                          'values': {'archive_batch_number': '305',
                                     'archive_revision_number': '6',
                                     'archive_storage_medium': 'Digital',
                                     'attachment_count': '5',
                                     'audit_reference_number': '1024',
                                     'document_page_count': '2',
                                     'document_template_family': 'Standard',
                                     'label_print_count': '2',
                                     'record_display_theme': 'Slate',
                                     'record_label_font': 'Calibri',
                                     'record_retention_months': '12',
                                     'record_view_count': '58',
                                     'supplier_id': 'S1',
                                     'transportation_cost_to_D1': '269.3910588020795',
                                     'transportation_cost_to_D2': '1.453733539093394',
                                     'transportation_cost_to_D3': '99.60345345756603',
                                     'transportation_cost_to_D4': '26.64078166309837',
                                     'transportation_cost_to_D5': '9.537688956880922'}},
                         {'source_row': 1,
                          'values': {'archive_batch_number': '303',
                                     'archive_revision_number': '1',
                                     'archive_storage_medium': 'Paper',
                                     'attachment_count': '3',
                                     'audit_reference_number': '1012',
                                     'document_page_count': '16',
                                     'document_template_family': 'Compact',
                                     'label_print_count': '2',
                                     'record_display_theme': 'Amber',
                                     'record_label_font': 'Arial',
                                     'record_retention_months': '48',
                                     'record_view_count': '27',
                                     'supplier_id': 'S2',
                                     'transportation_cost_to_D1': '9.291846876785185',
                                     'transportation_cost_to_D2': '10.874778437070225',
                                     'transportation_cost_to_D3': '144.52609291614627',
                                     'transportation_cost_to_D4': '11.420133077898234',
                                     'transportation_cost_to_D5': '153.1756819927813'}},
                         {'source_row': 2,
                          'values': {'archive_batch_number': '305',
                                     'archive_revision_number': '3',
                                     'archive_storage_medium': 'Paper',
                                     'attachment_count': '4',
                                     'audit_reference_number': '1060',
                                     'document_page_count': '6',
                                     'document_template_family': 'Landscape',
                                     'label_print_count': '6',
                                     'record_display_theme': 'Slate',
                                     'record_label_font': 'Calibri',
                                     'record_retention_months': '36',
                                     'record_view_count': '12',
                                     'supplier_id': 'S3',
                                     'transportation_cost_to_D1': '9.674584301671008',
                                     'transportation_cost_to_D2': '2.6191650959687944',
                                     'transportation_cost_to_D3': '100.8242249168735',
                                     'transportation_cost_to_D4': '3.212191088791688',
                                     'transportation_cost_to_D5': '133.8493396124168'}},
                         {'source_row': 3,
                          'values': {'archive_batch_number': '304',
                                     'archive_revision_number': '2',
                                     'archive_storage_medium': 'Digital',
                                     'attachment_count': '3',
                                     'audit_reference_number': '1060',
                                     'document_page_count': '16',
                                     'document_template_family': 'Compact',
                                     'label_print_count': '4',
                                     'record_display_theme': 'Azure',
                                     'record_label_font': 'Helvetica',
                                     'record_retention_months': '36',
                                     'record_view_count': '76',
                                     'supplier_id': 'S4',
                                     'transportation_cost_to_D1': '270.57498480010247',
                                     'transportation_cost_to_D2': '32.50253586',
                                     'transportation_cost_to_D3': '4.6842098096469815',
                                     'transportation_cost_to_D4': '1.5682269686546804',
                                     'transportation_cost_to_D5': '9.58927599'}},
                         {'source_row': 4,
                          'values': {'archive_batch_number': '304',
                                     'archive_revision_number': '6',
                                     'archive_storage_medium': 'Hybrid',
                                     'attachment_count': '4',
                                     'audit_reference_number': '1036',
                                     'document_page_count': '4',
                                     'document_template_family': 'Compact',
                                     'label_print_count': '2',
                                     'record_display_theme': 'Azure',
                                     'record_label_font': 'Helvetica',
                                     'record_retention_months': '36',
                                     'record_view_count': '43',
                                     'supplier_id': 'S5',
                                     'transportation_cost_to_D1': '226.0331910675782',
                                     'transportation_cost_to_D2': '8.669161980826471',
                                     'transportation_cost_to_D3': '65.47681316968448',
                                     'transportation_cost_to_D4': '9.068765258459958',
                                     'transportation_cost_to_D5': '202.65015316425533'}}],
             'returned_rows': 5,
             'role': 'file_2',
             'table_id': 'file_2_view_0'}],
 'validation': {'fallback_reason': "Matrix validation failed: {'matrix_table_id': 'file_2_view_0', 'shape': [5, 4], "
                                   "'expected_shape': [5, 5], 'row_ids_aligned': True, 'column_ids_aligned': False, "
                                   "'row_mapping_basis': 'exact', 'column_mapping_basis': 'unresolved'}",
                'planner_errors': ["Matrix validation failed: {'matrix_table_id': 'file_2_view_0', 'shape': [5, 4], "
                                   "'expected_shape': [5, 5], 'row_ids_aligned': True, 'column_ids_aligned': False, "
                                   "'row_mapping_basis': 'exact', 'column_mapping_basis': 'unresolved'}"],
                'status': 'FALLBACK_FULL_DATA'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    S = set()
    D = set()
    frame_supply = CSVQA_FRAMES['file_1_view_0']
    frame_cost = CSVQA_FRAMES['file_2_view_0']
    frame_demand = CSVQA_FRAMES['file_0_view_0']
    for (_, row) in frame_supply.iterrows():
        S.add(row['supplier_id'])
    for (_, row) in frame_cost.iterrows():
        S.add(row['supplier_id'])
    for (_, row) in frame_demand.iterrows():
        D.add(row['customer_id'])
    S = list(S)
    D = list(D)
    demand = {}
    for (_, row) in frame_demand.iterrows():
        j = row['customer_id']
        try:
            demand[j] = float(row['demand_units'])
        except Exception:
            raise ValueError(f"Invalid demand_units for customer_id {j}: {row['demand_units']}")
    supply_capacity = {}
    for (_, row) in frame_supply.iterrows():
        i = row['supplier_id']
        try:
            supply_capacity[i] = float(row['supply_capacity_units'])
        except Exception:
            raise ValueError(f"Invalid supply_capacity_units for supplier_id {i}: {row['supply_capacity_units']}")
    cost = {}
    for (_, row) in frame_cost.iterrows():
        i = row['supplier_id']
        cost[i] = {}
        for j in D:
            col = f'transportation_cost_to_{j}'
            if col not in row or row[col] == '' or row[col] is None:
                raise ValueError(f'Missing transportation cost for supplier_id {i} to customer_id {j}')
            try:
                cost[i][j] = float(row[col])
            except Exception:
                raise ValueError(f'Invalid transportation cost for supplier_id {i} to customer_id {j}: {row[col]}')
    for i in S:
        if i not in cost:
            raise ValueError(f'Missing cost row for supplier_id {i}')
        for j in D:
            if j not in cost[i]:
                raise ValueError(f'Missing cost for supplier_id {i} to customer_id {j}')
    for j in D:
        if j not in demand:
            raise ValueError(f'Missing demand for customer_id {j}')
    for i in S:
        if i not in supply_capacity:
            raise ValueError(f'Missing supply capacity for supplier_id {i}')
    m = gp.Model('GreenMart_Transportation')
    m.Params.MIPGap = 0.0001
    quantity_vars = m.addVars(S, D, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[i][j] * quantity_vars[i, j] for i in S for j in D)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((quantity_vars[i, j] for i in S)) >= demand[j] for j in D), name='')
    m.addConstrs((gp.quicksum((quantity_vars[i, j] for j in D)) <= supply_capacity[i] for i in S), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for variable in m.getVars():
            print(f'{variable.VarName}: {variable.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()