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
                         'customer_newsletter_open_count_2025_q4',
                         'customer_support_ticket_count',
                         'customer_product_inquiry_count_2025_q4',
                         'demand_units',
                         'customer_catalog_download_count_2025_q4'],
             'file_index': 0,
             'file_name': 'customer_demand.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0,
                          'values': {'customer_catalog_download_count_2025_q4': '0',
                                     'customer_id': 'D1',
                                     'customer_newsletter_open_count_2025_q4': '8',
                                     'customer_product_inquiry_count_2025_q4': '4',
                                     'customer_support_ticket_count': '12',
                                     'demand_units': '428'}},
                         {'source_row': 1,
                          'values': {'customer_catalog_download_count_2025_q4': '0',
                                     'customer_id': 'D2',
                                     'customer_newsletter_open_count_2025_q4': '8',
                                     'customer_product_inquiry_count_2025_q4': '2',
                                     'customer_support_ticket_count': '8',
                                     'demand_units': '217'}},
                         {'source_row': 2,
                          'values': {'customer_catalog_download_count_2025_q4': '4',
                                     'customer_id': 'D3',
                                     'customer_newsletter_open_count_2025_q4': '8',
                                     'customer_product_inquiry_count_2025_q4': '4',
                                     'customer_support_ticket_count': '5',
                                     'demand_units': '214'}},
                         {'source_row': 3,
                          'values': {'customer_catalog_download_count_2025_q4': '2',
                                     'customer_id': 'D4',
                                     'customer_newsletter_open_count_2025_q4': '1',
                                     'customer_product_inquiry_count_2025_q4': '2',
                                     'customer_support_ticket_count': '1',
                                     'demand_units': '380'}},
                         {'source_row': 4,
                          'values': {'customer_catalog_download_count_2025_q4': '1',
                                     'customer_id': 'D5',
                                     'customer_newsletter_open_count_2025_q4': '5',
                                     'customer_product_inquiry_count_2025_q4': '2',
                                     'customer_support_ticket_count': '17',
                                     'demand_units': '254'}}],
             'returned_rows': 5,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['supply_planning_meeting_count_2025_q4',
                         'fleet_vehicle_count',
                         'supplier_id',
                         'capacity_team_training_hours_2025_q4',
                         'production_handbook_revision_count_2025_q4',
                         'supply_capacity_units'],
             'file_index': 1,
             'file_name': 'supply_capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0,
                          'values': {'capacity_team_training_hours_2025_q4': '12',
                                     'fleet_vehicle_count': '5',
                                     'production_handbook_revision_count_2025_q4': '5',
                                     'supplier_id': 'S1',
                                     'supply_capacity_units': '428',
                                     'supply_planning_meeting_count_2025_q4': '3'}},
                         {'source_row': 1,
                          'values': {'capacity_team_training_hours_2025_q4': '16',
                                     'fleet_vehicle_count': '30',
                                     'production_handbook_revision_count_2025_q4': '3',
                                     'supplier_id': 'S2',
                                     'supply_capacity_units': '217',
                                     'supply_planning_meeting_count_2025_q4': '4'}},
                         {'source_row': 2,
                          'values': {'capacity_team_training_hours_2025_q4': '32',
                                     'fleet_vehicle_count': '5',
                                     'production_handbook_revision_count_2025_q4': '2',
                                     'supplier_id': 'S3',
                                     'supply_capacity_units': '214',
                                     'supply_planning_meeting_count_2025_q4': '4'}},
                         {'source_row': 3,
                          'values': {'capacity_team_training_hours_2025_q4': '32',
                                     'fleet_vehicle_count': '5',
                                     'production_handbook_revision_count_2025_q4': '5',
                                     'supplier_id': 'S4',
                                     'supply_capacity_units': '380',
                                     'supply_planning_meeting_count_2025_q4': '3'}},
                         {'source_row': 4,
                          'values': {'capacity_team_training_hours_2025_q4': '16',
                                     'fleet_vehicle_count': '30',
                                     'production_handbook_revision_count_2025_q4': '4',
                                     'supplier_id': 'S5',
                                     'supply_capacity_units': '254',
                                     'supply_planning_meeting_count_2025_q4': '6'}}],
             'returned_rows': 5,
             'role': 'file_1',
             'table_id': 'file_1_view_0'},
            {'columns': ['transport_training_format',
                         'supplier_id',
                         'customer_support_staff_count',
                         'dispatch_document_format',
                         'transportation_cost_to_D1',
                         'transportation_cost_to_D2',
                         'driver_training_session_count_2025_q4',
                         'dispatch_document_review_count_2025_q4',
                         'shipping_label_reprint_count_2025_q4',
                         'carrier_communication_channel',
                         'transportation_cost_to_D3',
                         'route_signage_inspection_count_2025_q4',
                         'operations_region',
                         'transportation_cost_to_D4',
                         'delivery_tracking_inquiry_count_2025_q4',
                         'transportation_cost_to_D5',
                         'carrier_coordination_meeting_count_2025_q4',
                         'annual_inspection_count'],
             'file_index': 2,
             'file_name': 'transportation_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0,
                          'values': {'annual_inspection_count': '1',
                                     'carrier_communication_channel': 'Email',
                                     'carrier_coordination_meeting_count_2025_q4': '3',
                                     'customer_support_staff_count': '8',
                                     'delivery_tracking_inquiry_count_2025_q4': '30',
                                     'dispatch_document_format': 'Paper',
                                     'dispatch_document_review_count_2025_q4': '45',
                                     'driver_training_session_count_2025_q4': '3',
                                     'operations_region': 'East',
                                     'route_signage_inspection_count_2025_q4': '6',
                                     'shipping_label_reprint_count_2025_q4': '7',
                                     'supplier_id': 'S1',
                                     'transport_training_format': 'Online',
                                     'transportation_cost_to_D1': '269.3910588020795',
                                     'transportation_cost_to_D2': '1.453733539093394',
                                     'transportation_cost_to_D3': '99.60345345756603',
                                     'transportation_cost_to_D4': '26.64078166309837',
                                     'transportation_cost_to_D5': '9.537688956880922'}},
                         {'source_row': 1,
                          'values': {'annual_inspection_count': '3',
                                     'carrier_communication_channel': 'Email',
                                     'carrier_coordination_meeting_count_2025_q4': '2',
                                     'customer_support_staff_count': '8',
                                     'delivery_tracking_inquiry_count_2025_q4': '10',
                                     'dispatch_document_format': 'Digital',
                                     'dispatch_document_review_count_2025_q4': '10',
                                     'driver_training_session_count_2025_q4': '2',
                                     'operations_region': 'South',
                                     'route_signage_inspection_count_2025_q4': '3',
                                     'shipping_label_reprint_count_2025_q4': '15',
                                     'supplier_id': 'S2',
                                     'transport_training_format': 'Online',
                                     'transportation_cost_to_D1': '9.291846876785185',
                                     'transportation_cost_to_D2': '10.874778437070225',
                                     'transportation_cost_to_D3': '144.52609291614627',
                                     'transportation_cost_to_D4': '11.420133077898234',
                                     'transportation_cost_to_D5': '153.1756819927813'}},
                         {'source_row': 2,
                          'values': {'annual_inspection_count': '6',
                                     'carrier_communication_channel': 'Phone',
                                     'carrier_coordination_meeting_count_2025_q4': '6',
                                     'customer_support_staff_count': '5',
                                     'delivery_tracking_inquiry_count_2025_q4': '5',
                                     'dispatch_document_format': 'Hybrid',
                                     'dispatch_document_review_count_2025_q4': '45',
                                     'driver_training_session_count_2025_q4': '1',
                                     'operations_region': 'South',
                                     'route_signage_inspection_count_2025_q4': '1',
                                     'shipping_label_reprint_count_2025_q4': '7',
                                     'supplier_id': 'S3',
                                     'transport_training_format': 'Classroom',
                                     'transportation_cost_to_D1': '9.674584301671008',
                                     'transportation_cost_to_D2': '2.6191650959687944',
                                     'transportation_cost_to_D3': '100.8242249168735',
                                     'transportation_cost_to_D4': '3.212191088791688',
                                     'transportation_cost_to_D5': '133.8493396124168'}},
                         {'source_row': 3,
                          'values': {'annual_inspection_count': '4',
                                     'carrier_communication_channel': 'Phone',
                                     'carrier_coordination_meeting_count_2025_q4': '8',
                                     'customer_support_staff_count': '16',
                                     'delivery_tracking_inquiry_count_2025_q4': '30',
                                     'dispatch_document_format': 'Paper',
                                     'dispatch_document_review_count_2025_q4': '15',
                                     'driver_training_session_count_2025_q4': '2',
                                     'operations_region': 'East',
                                     'route_signage_inspection_count_2025_q4': '3',
                                     'shipping_label_reprint_count_2025_q4': '10',
                                     'supplier_id': 'S4',
                                     'transport_training_format': 'Workshop',
                                     'transportation_cost_to_D1': '270.57498480010247',
                                     'transportation_cost_to_D2': '32.50253586',
                                     'transportation_cost_to_D3': '4.6842098096469815',
                                     'transportation_cost_to_D4': '1.5682269686546804',
                                     'transportation_cost_to_D5': '9.58927599'}},
                         {'source_row': 4,
                          'values': {'annual_inspection_count': '4',
                                     'carrier_communication_channel': 'Portal',
                                     'carrier_coordination_meeting_count_2025_q4': '6',
                                     'customer_support_staff_count': '20',
                                     'delivery_tracking_inquiry_count_2025_q4': '30',
                                     'dispatch_document_format': 'Hybrid',
                                     'dispatch_document_review_count_2025_q4': '45',
                                     'driver_training_session_count_2025_q4': '2',
                                     'operations_region': 'South',
                                     'route_signage_inspection_count_2025_q4': '2',
                                     'shipping_label_reprint_count_2025_q4': '10',
                                     'supplier_id': 'S5',
                                     'transport_training_format': 'Online',
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

def solve_problem(CSVQA_FRAMES):
    supply_frame = CSVQA_FRAMES['file_1_view_0']
    S = []
    supply_capacity = {}
    for (_, row) in supply_frame.iterrows():
        s = row['supplier_id']
        S.append(s)
        supply_capacity[s] = float(row['supply_capacity_units'])
    demand_frame = CSVQA_FRAMES['file_0_view_0']
    D = []
    demand = {}
    for (_, row) in demand_frame.iterrows():
        d = row['customer_id']
        D.append(d)
        demand[d] = float(row['demand_units'])
    cost_frame = CSVQA_FRAMES['file_2_view_0']
    cost = {}
    for (_, row) in cost_frame.iterrows():
        s = row['supplier_id']
        cost[s] = {}
        for d in D:
            col = f'transportation_cost_to_{d}'
            if col not in row or row[col] == '' or row[col] is None:
                raise ValueError(f'Missing transportation cost for supplier {s} to customer {d}')
            cost[s][d] = float(row[col])
    if set(cost.keys()) != set(S):
        raise ValueError('Mismatch in supplier IDs between cost matrix and supply_capacity')
    for s in S:
        if set(cost[s].keys()) != set(D):
            raise ValueError(f'Mismatch in customer IDs for supplier {s} in cost matrix')
    m = gp.Model('GreenMart_Transportation')
    m.Params.MIPGap = 0.0001
    quantity_vars = m.addVars(S, D, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[s][d] * quantity_vars[s, d] for s in S for d in D)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((quantity_vars[s, d] for s in S)) >= demand[d] for d in D), name='')
    m.addConstrs((gp.quicksum((quantity_vars[s, d] for d in D)) <= supply_capacity[s] for s in S), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem(CSVQA_FRAMES)