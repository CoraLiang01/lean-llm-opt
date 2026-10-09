CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'Amazon runs several distribution centers that deliver essential goods daily to different customer groups. '
          'The daily demand for each customer group is listed in ‚Äúcustomer_demand.csv,‚Äù while the daily supply '
          'capacity of the distribution centers is outlined in ‚Äúsupply_capacity.csv.‚Äù The transportation cost per '
          'unit from each distribution center to each customer group is specified in ‚Äútransportation_costs.csv.‚Äù '
          'The goal is to decide the quantity of goods to be shipped from each distribution center to each customer '
          'group, ensuring all demands are fulfilled without exceeding the supply capacity of any center, while '
          'minimizing the total transportation cost.',
 'relationships': [],
 'route': 'TP',
 'tables': [{'columns': ['customer_id', 'archive_revision_number', 'demand'],
             'file_index': 0,
             'file_name': 'customer_demand.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 8,
             'records': [{'source_row': 0,
                          'values': {'archive_revision_number': '4', 'customer_id': 'demand1', 'demand': '9'}},
                         {'source_row': 1,
                          'values': {'archive_revision_number': '4', 'customer_id': 'demand2', 'demand': '66'}},
                         {'source_row': 2,
                          'values': {'archive_revision_number': '3', 'customer_id': 'demand3', 'demand': '56'}},
                         {'source_row': 3,
                          'values': {'archive_revision_number': '6', 'customer_id': 'demand4', 'demand': '17'}},
                         {'source_row': 4,
                          'values': {'archive_revision_number': '3', 'customer_id': 'demand5', 'demand': '43'}},
                         {'source_row': 5,
                          'values': {'archive_revision_number': '4', 'customer_id': 'demand6', 'demand': '62'}},
                         {'source_row': 6,
                          'values': {'archive_revision_number': '4', 'customer_id': 'demand7', 'demand': '10'}},
                         {'source_row': 7,
                          'values': {'archive_revision_number': '4', 'customer_id': 'demand8', 'demand': '37'}}],
             'returned_rows': 8,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['archive_revision_number', 'supplier_id', 'supply_capacity'],
             'file_index': 1,
             'file_name': 'supply_capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 8,
             'records': [{'source_row': 0,
                          'values': {'archive_revision_number': '4',
                                     'supplier_id': 'supplier1',
                                     'supply_capacity': '60'}},
                         {'source_row': 1,
                          'values': {'archive_revision_number': '2',
                                     'supplier_id': 'supplier2',
                                     'supply_capacity': '22'}},
                         {'source_row': 2,
                          'values': {'archive_revision_number': '6',
                                     'supplier_id': 'supplier3',
                                     'supply_capacity': '16'}},
                         {'source_row': 3,
                          'values': {'archive_revision_number': '3',
                                     'supplier_id': 'supplier4',
                                     'supply_capacity': '14'}},
                         {'source_row': 4,
                          'values': {'archive_revision_number': '1',
                                     'supplier_id': 'supplier5',
                                     'supply_capacity': '19'}},
                         {'source_row': 5,
                          'values': {'archive_revision_number': '2',
                                     'supplier_id': 'supplier6',
                                     'supply_capacity': '70'}},
                         {'source_row': 6,
                          'values': {'archive_revision_number': '5',
                                     'supplier_id': 'supplier7',
                                     'supply_capacity': '60'}},
                         {'source_row': 7,
                          'values': {'archive_revision_number': '2',
                                     'supplier_id': 'supplier8',
                                     'supply_capacity': '39'}}],
             'returned_rows': 8,
             'role': 'file_1',
             'table_id': 'file_1_view_0'},
            {'columns': ['supplier_id',
                         'document_page_count',
                         'transportation_cost_to_demand1',
                         'record_view_count',
                         'transportation_cost_to_demand2',
                         'transportation_cost_to_demand3',
                         'record_display_theme',
                         'transportation_cost_to_demand4',
                         'archive_batch_number',
                         'transportation_cost_to_demand5',
                         'archive_revision_number',
                         'transportation_cost_to_demand6',
                         'transportation_cost_to_demand7',
                         'transportation_cost_to_demand8'],
             'file_index': 2,
             'file_name': 'transportation_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 8,
             'records': [{'source_row': 0,
                          'values': {'archive_batch_number': '301',
                                     'archive_revision_number': '6',
                                     'document_page_count': '2',
                                     'record_display_theme': 'Azure',
                                     'record_view_count': '76',
                                     'supplier_id': 'supply1',
                                     'transportation_cost_to_demand1': '0.03020736643461065',
                                     'transportation_cost_to_demand2': '229.50723504640203',
                                     'transportation_cost_to_demand3': '198.62356558205792',
                                     'transportation_cost_to_demand4': '12.995050640153751',
                                     'transportation_cost_to_demand5': '211.20732124396406',
                                     'transportation_cost_to_demand6': '134.9442985029274',
                                     'transportation_cost_to_demand7': '9.822206398831067',
                                     'transportation_cost_to_demand8': '11.394077543225675'}},
                         {'source_row': 1,
                          'values': {'archive_batch_number': '303',
                                     'archive_revision_number': '4',
                                     'document_page_count': '6',
                                     'record_display_theme': 'Azure',
                                     'record_view_count': '43',
                                     'supplier_id': 'supply2',
                                     'transportation_cost_to_demand1': '232.34691308087835',
                                     'transportation_cost_to_demand2': '3.6258726438627473',
                                     'transportation_cost_to_demand3': '0.28605434149404785',
                                     'transportation_cost_to_demand4': '45.73127693242935',
                                     'transportation_cost_to_demand5': '2.8304796563034573',
                                     'transportation_cost_to_demand6': '107.05891033185472',
                                     'transportation_cost_to_demand7': '299.96317913389305',
                                     'transportation_cost_to_demand8': '23.79935436307657'}},
                         {'source_row': 2,
                          'values': {'archive_batch_number': '303',
                                     'archive_revision_number': '6',
                                     'document_page_count': '6',
                                     'record_display_theme': 'Olive',
                                     'record_view_count': '43',
                                     'supplier_id': 'supply3',
                                     'transportation_cost_to_demand1': '11.061938334356302',
                                     'transportation_cost_to_demand2': '0.2041995326579051',
                                     'transportation_cost_to_demand3': '0.2789447278030927',
                                     'transportation_cost_to_demand4': '45.721912724349636',
                                     'transportation_cost_to_demand5': '59.54895565737313',
                                     'transportation_cost_to_demand6': '5.097536739581239',
                                     'transportation_cost_to_demand7': '300.00118415135785',
                                     'transportation_cost_to_demand8': '23.711282707746893'}},
                         {'source_row': 3,
                          'values': {'archive_batch_number': '305',
                                     'archive_revision_number': '2',
                                     'document_page_count': '6',
                                     'record_display_theme': 'Amber',
                                     'record_view_count': '91',
                                     'supplier_id': 'supply4',
                                     'transportation_cost_to_demand1': '235.1794835706472',
                                     'transportation_cost_to_demand2': '43.794668963036194',
                                     'transportation_cost_to_demand3': '40.709846782945924',
                                     'transportation_cost_to_demand4': '0.07774496620087613',
                                     'transportation_cost_to_demand5': '4.237728183419554',
                                     'transportation_cost_to_demand6': '131.70915517494691',
                                     'transportation_cost_to_demand7': '296.55587567706743',
                                     'transportation_cost_to_demand8': '29.810940017561297'}},
                         {'source_row': 4,
                          'values': {'archive_batch_number': '304',
                                     'archive_revision_number': '6',
                                     'document_page_count': '16',
                                     'record_display_theme': 'Olive',
                                     'record_view_count': '58',
                                     'supplier_id': 'supply5',
                                     'transportation_cost_to_demand1': '211.85808746383796',
                                     'transportation_cost_to_demand2': '47.60180876530328',
                                     'transportation_cost_to_demand3': '50.04007716193931',
                                     'transportation_cost_to_demand4': '86.14548807358399',
                                     'transportation_cost_to_demand5': '0.06197897916874956',
                                     'transportation_cost_to_demand6': '5.3345515296262205',
                                     'transportation_cost_to_demand7': '270.06290423798396',
                                     'transportation_cost_to_demand8': '3.853933133973331'}},
                         {'source_row': 5,
                          'values': {'archive_batch_number': '301',
                                     'archive_revision_number': '2',
                                     'document_page_count': '8',
                                     'record_display_theme': 'Amber',
                                     'record_view_count': '43',
                                     'supplier_id': 'supply6',
                                     'transportation_cost_to_demand1': '6.45506633554524',
                                     'transportation_cost_to_demand2': '88.16323623354015',
                                     'transportation_cost_to_demand3': '5.047091671641611',
                                     'transportation_cost_to_demand4': '151.46120287365497',
                                     'transportation_cost_to_demand5': '5.290760161059401',
                                     'transportation_cost_to_demand6': '0.04602205335871525',
                                     'transportation_cost_to_demand7': '9.93670660180487',
                                     'transportation_cost_to_demand8': '103.75460989446313'}},
                         {'source_row': 6,
                          'values': {'archive_batch_number': '303',
                                     'archive_revision_number': '6',
                                     'document_page_count': '12',
                                     'record_display_theme': 'Amber',
                                     'record_view_count': '58',
                                     'supplier_id': 'supply7',
                                     'transportation_cost_to_demand1': '174.27229047340035',
                                     'transportation_cost_to_demand2': '250.58223528739327',
                                     'transportation_cost_to_demand3': '253.90413041857263',
                                     'transportation_cost_to_demand4': '16.235467318386764',
                                     'transportation_cost_to_demand5': '12.643140514778086',
                                     'transportation_cost_to_demand6': '175.0672824108511',
                                     'transportation_cost_to_demand7': '2.983839625303656',
                                     'transportation_cost_to_demand8': '317.0655193866389'}},
                         {'source_row': 7,
                          'values': {'archive_batch_number': '302',
                                     'archive_revision_number': '2',
                                     'document_page_count': '6',
                                     'record_display_theme': 'Amber',
                                     'record_view_count': '58',
                                     'supplier_id': 'supply8',
                                     'transportation_cost_to_demand1': '207.87006253790491',
                                     'transportation_cost_to_demand2': '1.517168471518212',
                                     'transportation_cost_to_demand3': '24.027239288137153',
                                     'transportation_cost_to_demand4': '27.133999276450346',
                                     'transportation_cost_to_demand5': '73.20672468851855',
                                     'transportation_cost_to_demand6': '125.72910359893308',
                                     'transportation_cost_to_demand7': '15.463103251642147',
                                     'transportation_cost_to_demand8': '0.20164987511903337'}}],
             'returned_rows': 8,
             'role': 'file_2',
             'table_id': 'file_2_view_0'}],
 'validation': {'fallback_reason': "Matrix validation failed: {'matrix_table_id': 'file_2_view_0', 'shape': [8, 9], "
                                   "'expected_shape': [8, 8], 'row_ids_aligned': True, 'column_ids_aligned': False, "
                                   "'row_mapping_basis': 'unique_complete_suffix', 'column_mapping_basis': "
                                   "'unresolved'}",
                'planner_errors': ["Matrix validation failed: {'matrix_table_id': 'file_2_view_0', 'shape': [8, 9], "
                                   "'expected_shape': [8, 8], 'row_ids_aligned': True, 'column_ids_aligned': False, "
                                   "'row_mapping_basis': 'unique_complete_suffix', 'column_mapping_basis': "
                                   "'unresolved'}"],
                'status': 'FALLBACK_FULL_DATA'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem(CSVQA_FRAMES):
    supply_frame = CSVQA_FRAMES['file_1_view_0']
    suppliers = []
    supply_capacity = {}
    for (_, row) in supply_frame.iterrows():
        supplier_id = row['supplier_id']
        suppliers.append(supplier_id)
        try:
            supply_capacity[supplier_id] = float(row['supply_capacity'])
        except Exception:
            raise ValueError(f"Invalid supply_capacity for supplier {supplier_id}: {row['supply_capacity']}")
    demand_frame = CSVQA_FRAMES['file_0_view_0']
    customers = []
    demand = {}
    for (_, row) in demand_frame.iterrows():
        customer_id = row['customer_id']
        customers.append(customer_id)
        try:
            demand[customer_id] = float(row['demand'])
        except Exception:
            raise ValueError(f"Invalid demand for customer {customer_id}: {row['demand']}")
    cost_frame = CSVQA_FRAMES['file_2_view_0']
    cost = {}
    for (_, row) in cost_frame.iterrows():
        supplier_id = row['supplier_id']
        cost[supplier_id] = {}
        for customer_id in customers:
            cost_col = f'transportation_cost_to_{customer_id}'
            try:
                cost[supplier_id][customer_id] = float(row[cost_col])
            except Exception:
                raise ValueError(f'Invalid transportation cost for supplier {supplier_id} to customer {customer_id}: {row[cost_col]}')
    if set(suppliers) != set(cost.keys()):
        raise ValueError('Mismatch between suppliers in supply_capacity and transportation_costs.')
    for s in suppliers:
        if set(customers) != set(cost[s].keys()):
            raise ValueError(f'Mismatch between customers in transportation_costs for supplier {s}.')
    m = gp.Model('Amazon_Distribution_Transportation')
    m.Params.MIPGap = 0.0001
    quantity_vars = m.addVars(suppliers, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
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