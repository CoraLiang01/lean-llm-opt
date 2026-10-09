CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'There are multiple suppliers responsible for delivering essential goods daily to customer groups located in '
          'different regions. Each supplier has a specific daily supply capacity, detailed in "supply_capacity.csv", '
          'while the daily demand of each customer group is recorded in "customer_demand.csv". The transportation cost '
          'per unit of goods from each supplier to each customer group is provided in "transportation_costs.csv". The '
          'objective is to determine the optimal transportation plan, specifying how goods should be allocated from '
          'each supplier to each customer group, ensuring that all demands are met without exceeding the supply '
          'capacity of any supplier, while minimizing the total transportation cost.',
 'relationships': [],
 'route': 'TP',
 'tables': [{'columns': ['customer_id', 'archive_revision_number', 'demand'],
             'file_index': 0,
             'file_name': 'customer_demand.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0,
                          'values': {'archive_revision_number': '2', 'customer_id': 'C1', 'demand': '216'}},
                         {'source_row': 1,
                          'values': {'archive_revision_number': '3', 'customer_id': 'C2', 'demand': '168'}},
                         {'source_row': 2,
                          'values': {'archive_revision_number': '2', 'customer_id': 'C3', 'demand': '264'}},
                         {'source_row': 3,
                          'values': {'archive_revision_number': '2', 'customer_id': 'C4', 'demand': '216'}},
                         {'source_row': 4,
                          'values': {'archive_revision_number': '3', 'customer_id': 'C5', 'demand': '216'}},
                         {'source_row': 5,
                          'values': {'archive_revision_number': '4', 'customer_id': 'C6', 'demand': '192'}},
                         {'source_row': 6,
                          'values': {'archive_revision_number': '6', 'customer_id': 'C7', 'demand': '144'}},
                         {'source_row': 7,
                          'values': {'archive_revision_number': '5', 'customer_id': 'C8', 'demand': '168'}},
                         {'source_row': 8,
                          'values': {'archive_revision_number': '1', 'customer_id': 'C9', 'demand': '168'}},
                         {'source_row': 9,
                          'values': {'archive_revision_number': '2', 'customer_id': 'C10', 'demand': '168'}}],
             'returned_rows': 10,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['supplier_id', 'archive_revision_number', 'supply_capacity'],
             'file_index': 1,
             'file_name': 'supply_capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0,
                          'values': {'archive_revision_number': '1', 'supplier_id': 'S1', 'supply_capacity': '288'}},
                         {'source_row': 1,
                          'values': {'archive_revision_number': '2', 'supplier_id': 'S2', 'supply_capacity': '288'}},
                         {'source_row': 2,
                          'values': {'archive_revision_number': '5', 'supplier_id': 'S3', 'supply_capacity': '264'}},
                         {'source_row': 3,
                          'values': {'archive_revision_number': '2', 'supplier_id': 'S4', 'supply_capacity': '264'}},
                         {'source_row': 4,
                          'values': {'archive_revision_number': '5', 'supplier_id': 'S5', 'supply_capacity': '216'}},
                         {'source_row': 5,
                          'values': {'archive_revision_number': '5', 'supplier_id': 'S6', 'supply_capacity': '216'}},
                         {'source_row': 6,
                          'values': {'archive_revision_number': '2', 'supplier_id': 'S7', 'supply_capacity': '168'}},
                         {'source_row': 7,
                          'values': {'archive_revision_number': '2', 'supplier_id': 'S8', 'supply_capacity': '216'}},
                         {'source_row': 8,
                          'values': {'archive_revision_number': '6', 'supplier_id': 'S9', 'supply_capacity': '240'}},
                         {'source_row': 9,
                          'values': {'archive_revision_number': '1', 'supplier_id': 'S10', 'supply_capacity': '168'}}],
             'returned_rows': 10,
             'role': 'file_1',
             'table_id': 'file_1_view_0'},
            {'columns': ['attachment_count',
                         'supplier_id',
                         'transportation_cost_to_C1',
                         'transportation_cost_to_C2',
                         'transportation_cost_to_C3',
                         'document_page_count',
                         'transportation_cost_to_C4',
                         'transportation_cost_to_C5',
                         'transportation_cost_to_C6',
                         'record_view_count',
                         'record_display_theme',
                         'transportation_cost_to_C7',
                         'transportation_cost_to_C8',
                         'transportation_cost_to_C9',
                         'archive_batch_number',
                         'transportation_cost_to_C10',
                         'archive_revision_number'],
             'file_index': 2,
             'file_name': 'transportation_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0,
                          'values': {'archive_batch_number': '303',
                                     'archive_revision_number': '4',
                                     'attachment_count': '5',
                                     'document_page_count': '16',
                                     'record_display_theme': 'Amber',
                                     'record_view_count': '43',
                                     'supplier_id': 'S1',
                                     'transportation_cost_to_C1': '590.3648137',
                                     'transportation_cost_to_C10': '58.89547392',
                                     'transportation_cost_to_C2': '23.66917261',
                                     'transportation_cost_to_C3': '88.8900587',
                                     'transportation_cost_to_C4': '497.5222881',
                                     'transportation_cost_to_C5': '466.0903432',
                                     'transportation_cost_to_C6': '29.02209683',
                                     'transportation_cost_to_C7': '23.67524483',
                                     'transportation_cost_to_C8': '23.67776029',
                                     'transportation_cost_to_C9': '0.311839491'}},
                         {'source_row': 1,
                          'values': {'archive_batch_number': '302',
                                     'archive_revision_number': '3',
                                     'attachment_count': '5',
                                     'document_page_count': '2',
                                     'record_display_theme': 'Amber',
                                     'record_view_count': '27',
                                     'supplier_id': 'S2',
                                     'transportation_cost_to_C1': '2042.0715',
                                     'transportation_cost_to_C10': '67.29170751',
                                     'transportation_cost_to_C2': '2133.978484',
                                     'transportation_cost_to_C3': '705.1591203',
                                     'transportation_cost_to_C4': '101.5945452',
                                     'transportation_cost_to_C5': '2052.937657',
                                     'transportation_cost_to_C6': '1738.754951',
                                     'transportation_cost_to_C7': '101.6109497',
                                     'transportation_cost_to_C8': '101.6106217',
                                     'transportation_cost_to_C9': '122.4521427'}},
                         {'source_row': 2,
                          'values': {'archive_batch_number': '301',
                                     'archive_revision_number': '1',
                                     'attachment_count': '5',
                                     'document_page_count': '4',
                                     'record_display_theme': 'Azure',
                                     'record_view_count': '58',
                                     'supplier_id': 'S3',
                                     'transportation_cost_to_C1': '22.29722216',
                                     'transportation_cost_to_C10': '1008.671739',
                                     'transportation_cost_to_C2': '497.9271939',
                                     'transportation_cost_to_C3': '1653.082886',
                                     'transportation_cost_to_C4': '23.68545123',
                                     'transportation_cost_to_C5': '1386.080789',
                                     'transportation_cost_to_C6': '26.13715281',
                                     'transportation_cost_to_C7': '497.6220483',
                                     'transportation_cost_to_C8': '498.0935847',
                                     'transportation_cost_to_C9': '865.3816296'}},
                         {'source_row': 3,
                          'values': {'archive_batch_number': '305',
                                     'archive_revision_number': '1',
                                     'attachment_count': '3',
                                     'document_page_count': '8',
                                     'record_display_theme': 'Olive',
                                     'record_view_count': '76',
                                     'supplier_id': 'S4',
                                     'transportation_cost_to_C1': '960.7814534',
                                     'transportation_cost_to_C10': '1351.818907',
                                     'transportation_cost_to_C2': '49.12830005',
                                     'transportation_cost_to_C3': '1324.238697',
                                     'transportation_cost_to_C4': '1032.209548',
                                     'transportation_cost_to_C5': '0.078047254',
                                     'transportation_cost_to_C6': '53.30826873',
                                     'transportation_cost_to_C7': '49.13641672',
                                     'transportation_cost_to_C8': '1031.821442',
                                     'transportation_cost_to_C9': '466.0049531'}},
                         {'source_row': 4,
                          'values': {'archive_batch_number': '303',
                                     'archive_revision_number': '5',
                                     'attachment_count': '4',
                                     'document_page_count': '12',
                                     'record_display_theme': 'Slate',
                                     'record_view_count': '27',
                                     'supplier_id': 'S5',
                                     'transportation_cost_to_C1': '1471.272167',
                                     'transportation_cost_to_C10': '1094.669596',
                                     'transportation_cost_to_C2': '85.69560728',
                                     'transportation_cost_to_C3': '38.89266824',
                                     'transportation_cost_to_C4': '1542.050036',
                                     'transportation_cost_to_C5': '112.2051437',
                                     'transportation_cost_to_C6': '82.37020164',
                                     'transportation_cost_to_C7': '1542.33992',
                                     'transportation_cost_to_C8': '85.69238746',
                                     'transportation_cost_to_C9': '1924.936077'}},
                         {'source_row': 5,
                          'values': {'archive_batch_number': '304',
                                     'archive_revision_number': '5',
                                     'attachment_count': '2',
                                     'document_page_count': '4',
                                     'record_display_theme': 'Slate',
                                     'record_view_count': '27',
                                     'supplier_id': 'S6',
                                     'transportation_cost_to_C1': '191.9058726',
                                     'transportation_cost_to_C10': '929.8071683',
                                     'transportation_cost_to_C2': '158.5040103',
                                     'transportation_cost_to_C3': '91.0204535',
                                     'transportation_cost_to_C4': '184.447472',
                                     'transportation_cost_to_C5': '968.1467987',
                                     'transportation_cost_to_C6': '284.1076062',
                                     'transportation_cost_to_C7': '8.791061588',
                                     'transportation_cost_to_C8': '158.7052384',
                                     'transportation_cost_to_C9': '27.94387435'}},
                         {'source_row': 6,
                          'values': {'archive_batch_number': '302',
                                     'archive_revision_number': '3',
                                     'attachment_count': '2',
                                     'document_page_count': '12',
                                     'record_display_theme': 'Slate',
                                     'record_view_count': '91',
                                     'supplier_id': 'S7',
                                     'transportation_cost_to_C1': '81.23891457',
                                     'transportation_cost_to_C10': '849.9799407',
                                     'transportation_cost_to_C2': '0.374464222',
                                     'transportation_cost_to_C3': '2079.466865',
                                     'transportation_cost_to_C4': '0.306567176',
                                     'transportation_cost_to_C5': '1031.777296',
                                     'transportation_cost_to_C6': '7.203964492',
                                     'transportation_cost_to_C7': '0.076230722',
                                     'transportation_cost_to_C8': '0.03247388',
                                     'transportation_cost_to_C9': '23.68582797'}},
                         {'source_row': 7,
                          'values': {'archive_batch_number': '304',
                                     'archive_revision_number': '5',
                                     'attachment_count': '4',
                                     'document_page_count': '12',
                                     'record_display_theme': 'Slate',
                                     'record_view_count': '27',
                                     'supplier_id': 'S8',
                                     'transportation_cost_to_C1': '56.09931097',
                                     'transportation_cost_to_C10': '1348.836615',
                                     'transportation_cost_to_C2': '935.6143109',
                                     'transportation_cost_to_C3': '73.08824617',
                                     'transportation_cost_to_C4': '52.00392409',
                                     'transportation_cost_to_C5': '4.025792389',
                                     'transportation_cost_to_C6': '1002.232766',
                                     'transportation_cost_to_C7': '935.776603',
                                     'transportation_cost_to_C8': '935.7007252',
                                     'transportation_cost_to_C9': '612.8698719'}},
                         {'source_row': 8,
                          'values': {'archive_batch_number': '302',
                                     'archive_revision_number': '6',
                                     'attachment_count': '4',
                                     'document_page_count': '12',
                                     'record_display_theme': 'Amber',
                                     'record_view_count': '12',
                                     'supplier_id': 'S9',
                                     'transportation_cost_to_C1': '4.502283327',
                                     'transportation_cost_to_C10': '40.46575555',
                                     'transportation_cost_to_C2': '0.389958534',
                                     'transportation_cost_to_C3': '1782.466218',
                                     'transportation_cost_to_C4': '0.006345907',
                                     'transportation_cost_to_C5': '1031.991011',
                                     'transportation_cost_to_C6': '129.5066562',
                                     'transportation_cost_to_C7': '0.211831957',
                                     'transportation_cost_to_C8': '0.645730107',
                                     'transportation_cost_to_C9': '497.6272391'}},
                         {'source_row': 9,
                          'values': {'archive_batch_number': '305',
                                     'archive_revision_number': '2',
                                     'attachment_count': '3',
                                     'document_page_count': '6',
                                     'record_display_theme': 'Olive',
                                     'record_view_count': '76',
                                     'supplier_id': 'S10',
                                     'transportation_cost_to_C1': '333.686927',
                                     'transportation_cost_to_C10': '941.7526366',
                                     'transportation_cost_to_C2': '277.4719386',
                                     'transportation_cost_to_C3': '86.02096892',
                                     'transportation_cost_to_C4': '277.3083661',
                                     'transportation_cost_to_C5': '1004.464908',
                                     'transportation_cost_to_C6': '19.95033682',
                                     'transportation_cost_to_C7': '13.20207369',
                                     'transportation_cost_to_C8': '238.1432152',
                                     'transportation_cost_to_C9': '411.0580332'}}],
             'returned_rows': 10,
             'role': 'file_2',
             'table_id': 'file_2_view_0'}],
 'validation': {'fallback_reason': "Matrix validation failed: {'matrix_table_id': 'file_2_view_0', 'shape': [10, 11], "
                                   "'expected_shape': [10, 10], 'row_ids_aligned': True, 'column_ids_aligned': False, "
                                   "'row_mapping_basis': 'exact', 'column_mapping_basis': 'unresolved'}",
                'planner_errors': ["Matrix validation failed: {'matrix_table_id': 'file_2_view_0', 'shape': [10, 11], "
                                   "'expected_shape': [10, 10], 'row_ids_aligned': True, 'column_ids_aligned': False, "
                                   "'row_mapping_basis': 'exact', 'column_mapping_basis': 'unresolved'}"],
                'status': 'FALLBACK_FULL_DATA'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem(CSVQA_FRAMES):
    customer_df = CSVQA_FRAMES['file_0_view_0']
    supplier_df = CSVQA_FRAMES['file_1_view_0']
    cost_df = CSVQA_FRAMES['file_2_view_0']
    customers = []
    demand = {}
    for (_, row) in customer_df.iterrows():
        cid = row['customer_id']
        customers.append(cid)
        try:
            demand[cid] = float(row['demand'])
        except Exception:
            raise ValueError(f"Invalid demand for customer {cid}: {row['demand']}")
    suppliers = []
    supply_capacity = {}
    for (_, row) in supplier_df.iterrows():
        sid = row['supplier_id']
        suppliers.append(sid)
        try:
            supply_capacity[sid] = float(row['supply_capacity'])
        except Exception:
            raise ValueError(f"Invalid supply_capacity for supplier {sid}: {row['supply_capacity']}")
    cost = {}
    for (_, row) in cost_df.iterrows():
        sid = row['supplier_id']
        cost[sid] = {}
        for cid in customers:
            col = f'transportation_cost_to_{cid}'
            if col not in cost_df.columns:
                raise ValueError(f'Missing cost column {col} in transportation_costs.csv')
            try:
                cost[sid][cid] = float(row[col])
            except Exception:
                raise ValueError(f'Invalid cost for supplier {sid} to customer {cid}: {row[col]}')
    if set(cost.keys()) != set(suppliers):
        raise ValueError('Mismatch in supplier IDs between cost matrix and supplier list')
    for sid in suppliers:
        if set(cost[sid].keys()) != set(customers):
            raise ValueError(f'Mismatch in customer IDs for supplier {sid} in cost matrix')
    m = gp.Model('TP')
    m.Params.MIPGap = 0.0001
    quantity_keys = [(i, j) for i in suppliers for j in customers]
    quantity_vars = m.addVars(quantity_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[i][j] * quantity_vars[i, j] for i in suppliers for j in customers)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((quantity_vars[i, j] for i in suppliers)) >= demand[j] for j in customers), name='')
    m.addConstrs((gp.quicksum((quantity_vars[i, j] for j in customers)) <= supply_capacity[i] for i in suppliers), name='')
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')