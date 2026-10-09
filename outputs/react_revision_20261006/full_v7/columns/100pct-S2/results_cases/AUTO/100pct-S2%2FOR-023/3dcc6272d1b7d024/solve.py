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
 'tables': [{'columns': ['customer_support_ticket_count',
                         'Customer',
                         'demand',
                         'customer_newsletter_open_count_2025_q4'],
             'file_index': 0,
             'file_name': 'demand.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0,
                          'values': {'Customer': 'Customer_1',
                                     'customer_newsletter_open_count_2025_q4': '3',
                                     'customer_support_ticket_count': '3',
                                     'demand': '2397'}},
                         {'source_row': 1,
                          'values': {'Customer': 'Customer_2',
                                     'customer_newsletter_open_count_2025_q4': '0',
                                     'customer_support_ticket_count': '8',
                                     'demand': '1889'}},
                         {'source_row': 2,
                          'values': {'Customer': 'Customer_3',
                                     'customer_newsletter_open_count_2025_q4': '3',
                                     'customer_support_ticket_count': '1',
                                     'demand': '2518'}},
                         {'source_row': 3,
                          'values': {'Customer': 'Customer_4',
                                     'customer_newsletter_open_count_2025_q4': '3',
                                     'customer_support_ticket_count': '8',
                                     'demand': '3218'}},
                         {'source_row': 4,
                          'values': {'Customer': 'Customer_5',
                                     'customer_newsletter_open_count_2025_q4': '3',
                                     'customer_support_ticket_count': '5',
                                     'demand': '1813'}}],
             'returned_rows': 5,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['facility_reception_desk_count_2025_q4', 'facility_staff_count', 'Unnamed: 2', 'fixed_costs'],
             'file_index': 1,
             'file_name': 'fixed_cost.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0,
                          'values': {'Unnamed: 2': 'MOUNT AYR',
                                     'facility_reception_desk_count_2025_q4': '3',
                                     'facility_staff_count': '12',
                                     'fixed_costs': '96.58'}},
                         {'source_row': 1,
                          'values': {'Unnamed: 2': 'WAUKEE',
                                     'facility_reception_desk_count_2025_q4': '3',
                                     'facility_staff_count': '12',
                                     'fixed_costs': '94.06'}},
                         {'source_row': 2,
                          'values': {'Unnamed: 2': 'WAVERLY',
                                     'facility_reception_desk_count_2025_q4': '2',
                                     'facility_staff_count': '20',
                                     'fixed_costs': '94.37'}},
                         {'source_row': 3,
                          'values': {'Unnamed: 2': 'PELLA',
                                     'facility_reception_desk_count_2025_q4': '2',
                                     'facility_staff_count': '8',
                                     'fixed_costs': '82.88'}},
                         {'source_row': 4,
                          'values': {'Unnamed: 2': 'DES MOINES',
                                     'facility_reception_desk_count_2025_q4': '3',
                                     'facility_staff_count': '50',
                                     'fixed_costs': '94.95999999999999'}}],
             'returned_rows': 5,
             'role': 'file_1',
             'table_id': 'file_1_view_0'},
            {'columns': ['carrier_communication_channel',
                         'carrier_coordination_meeting_count_2025_q4',
                         'Unnamed: 2',
                         'CLARINDA',
                         'customer_support_staff_count',
                         'operations_region',
                         'dispatch_document_review_count_2025_q4',
                         'FORT MADISON',
                         'annual_inspection_count',
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
                                     'annual_inspection_count': '6',
                                     'carrier_communication_channel': 'Email',
                                     'carrier_coordination_meeting_count_2025_q4': '6',
                                     'customer_support_staff_count': '20',
                                     'dispatch_document_review_count_2025_q4': '10',
                                     'operations_region': 'West'}},
                         {'source_row': 1,
                          'values': {'BANCROFT': '90.69',
                                     'CLARINDA': '15.13',
                                     'FORT MADISON': '1.5',
                                     'SIOUX CITY': '1.43',
                                     'TOLEDO': '27.88',
                                     'Unnamed: 2': 'WAUKEE',
                                     'annual_inspection_count': '2',
                                     'carrier_communication_channel': 'Email',
                                     'carrier_coordination_meeting_count_2025_q4': '6',
                                     'customer_support_staff_count': '12',
                                     'dispatch_document_review_count_2025_q4': '10',
                                     'operations_region': 'South'}},
                         {'source_row': 2,
                          'values': {'BANCROFT': '78.73',
                                     'CLARINDA': '2.34',
                                     'FORT MADISON': '349.34',
                                     'SIOUX CITY': '246.6',
                                     'TOLEDO': '41.3',
                                     'Unnamed: 2': 'WAVERLY',
                                     'annual_inspection_count': '3',
                                     'carrier_communication_channel': 'Portal',
                                     'carrier_coordination_meeting_count_2025_q4': '2',
                                     'customer_support_staff_count': '16',
                                     'dispatch_document_review_count_2025_q4': '30',
                                     'operations_region': 'South'}},
                         {'source_row': 3,
                          'values': {'BANCROFT': '38.93',
                                     'CLARINDA': '1181.6',
                                     'FORT MADISON': '1458.53',
                                     'SIOUX CITY': '1646.36',
                                     'TOLEDO': '1924.55',
                                     'Unnamed: 2': 'PELLA',
                                     'annual_inspection_count': '3',
                                     'carrier_communication_channel': 'Portal',
                                     'carrier_coordination_meeting_count_2025_q4': '6',
                                     'customer_support_staff_count': '5',
                                     'dispatch_document_review_count_2025_q4': '20',
                                     'operations_region': 'North'}},
                         {'source_row': 4,
                          'values': {'BANCROFT': '103.84',
                                     'CLARINDA': '1030.8',
                                     'FORT MADISON': '43.48',
                                     'SIOUX CITY': '932.4299999999999',
                                     'TOLEDO': '55.39',
                                     'Unnamed: 2': 'DES MOINES',
                                     'annual_inspection_count': '3',
                                     'carrier_communication_channel': 'Email',
                                     'carrier_coordination_meeting_count_2025_q4': '8',
                                     'customer_support_staff_count': '16',
                                     'dispatch_document_review_count_2025_q4': '15',
                                     'operations_region': 'North'}}],
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
    demand_df = CSVQA_FRAMES['file_0_view_0']
    fixed_cost_df = CSVQA_FRAMES['file_1_view_0']
    trans_cost_df = CSVQA_FRAMES['file_2_view_0']
    I = [row['Unnamed: 2'] for row in fixed_cost_df.to_dict(orient='records')]
    J = [row['Customer'] for row in demand_df.to_dict(orient='records')]
    demand = {}
    for row in demand_df.to_dict(orient='records'):
        cust = row['Customer']
        try:
            demand[cust] = float(row['demand'])
        except Exception:
            raise ValueError(f"Invalid demand for customer {cust}: {row['demand']}")
    fixed_cost = {}
    for row in fixed_cost_df.to_dict(orient='records'):
        fac = row['Unnamed: 2']
        try:
            fixed_cost[fac] = float(row['fixed_costs'])
        except Exception:
            raise ValueError(f"Invalid fixed cost for facility {fac}: {row['fixed_costs']}")
    cost = {}
    for row in trans_cost_df.to_dict(orient='records'):
        supplier = row['Unnamed: 2']
        cost[supplier] = {}
        for store in J:
            if store not in row:
                raise ValueError(f'Store {store} not found in transportation_costs.csv columns')
            try:
                cost[supplier][store] = float(row[store])
            except Exception:
                raise ValueError(f'Invalid transportation cost for ({supplier},{store}): {row[store]}')
    D = {j: demand[j] for j in J}
    for i in I:
        if i not in fixed_cost:
            raise ValueError(f'Supplier {i} missing fixed cost')
        if i not in cost:
            raise ValueError(f'Supplier {i} missing in transportation cost matrix')
        for j in J:
            if j not in cost[i]:
                raise ValueError(f'Cost for ({i},{j}) missing in transportation cost matrix')
    for j in J:
        if j not in demand:
            raise ValueError(f'Store {j} missing demand')
    m = gp.Model('Iowa_FLP')
    m.setParam('MIPGap', 0.0001)
    quantity_keys = [(i, j) for i in I for j in J]
    quantity_vars = m.addVars(quantity_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    open_vars = m.addVars(I, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[i][j] * quantity_vars[i, j] for i in I for j in J)) + gp.quicksum((fixed_cost[i] * open_vars[i] for i in I)), GRB.MINIMIZE)
    for j in J:
        m.addConstr(gp.quicksum((quantity_vars[i, j] for i in I)) == demand[j], name=f'demand_{j}')
    for i in I:
        for j in J:
            m.addConstr(quantity_vars[i, j] <= D[j] * open_vars[i], name=f'activation_{i}_{j}')
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')