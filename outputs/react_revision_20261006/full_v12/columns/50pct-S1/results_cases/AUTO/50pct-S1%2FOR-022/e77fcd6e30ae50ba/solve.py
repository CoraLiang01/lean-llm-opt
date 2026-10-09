CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'In the Superstore chain, multiple branches require inventory replenishment, and several suppliers located '
          'in different cities can provide the necessary goods. Each supplier incurs a fixed cost upon starting '
          'operations, with the fixed cost data provided in the ‚Äúfixed_cost.csv‚Äù file. Each branch needs to source '
          'a certain quantity of goods from these suppliers. For each branch, the transportation cost per unit of '
          'goods from each supplier is recorded in the ‚Äútransportation_costs.csv‚Äù file. Demand information can be '
          "gained in 'demand.csv'. The objective is to determine which suppliers to activate so that the demand of all "
          'branches is met while minimizing the total cost. The decision variables y_i are binary, indicating whether '
          'a supplier is operational (open). The decision variables x_{ij} represent the quantity of goods that branch '
          'S_j sources from supplier F_i. For each branch, x_{ij} represents the proportion of the total supply '
          'obtained from different suppliers. These decision variables help determine the optimal allocation of supply '
          'to minimize the total of fixed and transportation costs.',
 'relationships': [],
 'route': 'FLP',
 'tables': [{'columns': ['customer_id', 'archive_revision_number', 'demand_units'],
             'file_index': 0,
             'file_name': 'demand.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0,
                          'values': {'archive_revision_number': '6', 'customer_id': 'C1', 'demand_units': '143'}},
                         {'source_row': 1,
                          'values': {'archive_revision_number': '5', 'customer_id': 'C2', 'demand_units': '6'}},
                         {'source_row': 2,
                          'values': {'archive_revision_number': '3', 'customer_id': 'C3', 'demand_units': '10'}},
                         {'source_row': 3,
                          'values': {'archive_revision_number': '1', 'customer_id': 'C4', 'demand_units': '25'}},
                         {'source_row': 4,
                          'values': {'archive_revision_number': '1', 'customer_id': 'C5', 'demand_units': '3'}}],
             'returned_rows': 5,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['facility_id', 'archive_revision_number', 'fixed_opening_cost'],
             'file_index': 1,
             'file_name': 'fixed_cost.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0,
                          'values': {'archive_revision_number': '3',
                                     'facility_id': 'S1',
                                     'fixed_opening_cost': '97.65'}},
                         {'source_row': 1,
                          'values': {'archive_revision_number': '5',
                                     'facility_id': 'S2',
                                     'fixed_opening_cost': '99.76'}},
                         {'source_row': 2,
                          'values': {'archive_revision_number': '1',
                                     'facility_id': 'S3',
                                     'fixed_opening_cost': '100.76'}},
                         {'source_row': 3,
                          'values': {'archive_revision_number': '2',
                                     'facility_id': 'S4',
                                     'fixed_opening_cost': '105.32'}},
                         {'source_row': 4,
                          'values': {'archive_revision_number': '5',
                                     'facility_id': 'S5',
                                     'fixed_opening_cost': '98.88'}}],
             'returned_rows': 5,
             'role': 'file_1',
             'table_id': 'file_1_view_0'},
            {'columns': ['facility_id',
                         'transportation_cost_to_C1',
                         'transportation_cost_to_C2',
                         'archive_revision_number',
                         'document_page_count',
                         'transportation_cost_to_C3',
                         'transportation_cost_to_C4',
                         'transportation_cost_to_C5',
                         'record_display_theme'],
             'file_index': 2,
             'file_name': 'transportation_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0,
                          'values': {'archive_revision_number': '2',
                                     'document_page_count': '16',
                                     'facility_id': 'S1',
                                     'record_display_theme': 'Azure',
                                     'transportation_cost_to_C1': '150.74',
                                     'transportation_cost_to_C2': '0.02',
                                     'transportation_cost_to_C3': '49.13',
                                     'transportation_cost_to_C4': '2080.15',
                                     'transportation_cost_to_C5': '426.4'}},
                         {'source_row': 1,
                          'values': {'archive_revision_number': '2',
                                     'document_page_count': '4',
                                     'facility_id': 'S2',
                                     'record_display_theme': 'Slate',
                                     'transportation_cost_to_C1': '233.05',
                                     'transportation_cost_to_C2': '97.73',
                                     'transportation_cost_to_C3': '49.84',
                                     'transportation_cost_to_C4': '1982.39',
                                     'transportation_cost_to_C5': '23.96'}},
                         {'source_row': 2,
                          'values': {'archive_revision_number': '2',
                                     'document_page_count': '12',
                                     'facility_id': 'S3',
                                     'record_display_theme': 'Olive',
                                     'transportation_cost_to_C1': '55.68',
                                     'transportation_cost_to_C2': '935.61',
                                     'transportation_cost_to_C3': '4.03',
                                     'transportation_cost_to_C4': '73.09',
                                     'transportation_cost_to_C5': '525.32'}},
                         {'source_row': 3,
                          'values': {'archive_revision_number': '4',
                                     'document_page_count': '8',
                                     'facility_id': 'S4',
                                     'record_display_theme': 'Slate',
                                     'transportation_cost_to_C1': '1483.82',
                                     'transportation_cost_to_C2': '1801.08',
                                     'transportation_cost_to_C3': '112.16',
                                     'transportation_cost_to_C4': '816.05',
                                     'transportation_cost_to_C5': '107.01'}},
                         {'source_row': 4,
                          'values': {'archive_revision_number': '5',
                                     'document_page_count': '2',
                                     'facility_id': 'S5',
                                     'record_display_theme': 'Olive',
                                     'transportation_cost_to_C1': '1119.47',
                                     'transportation_cost_to_C2': '884.31',
                                     'transportation_cost_to_C3': '0.08',
                                     'transportation_cost_to_C4': '1544.95',
                                     'transportation_cost_to_C5': '543.67'}}],
             'returned_rows': 5,
             'role': 'file_2',
             'table_id': 'file_2_view_0'}],
 'validation': {'fallback_reason': "Matrix validation failed: {'matrix_table_id': 'file_2_view_0', 'shape': [5, 6], "
                                   "'expected_shape': [5, 5], 'row_ids_aligned': True, 'column_ids_aligned': False, "
                                   "'row_mapping_basis': 'exact', 'column_mapping_basis': 'unresolved'}",
                'planner_errors': ["Matrix validation failed: {'matrix_table_id': 'file_2_view_0', 'shape': [5, 6], "
                                   "'expected_shape': [5, 5], 'row_ids_aligned': True, 'column_ids_aligned': False, "
                                   "'row_mapping_basis': 'exact', 'column_mapping_basis': 'unresolved'}"],
                'status': 'FALLBACK_FULL_DATA'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem(CSVQA_FRAMES):
    demand_frame = CSVQA_FRAMES['file_0_view_0']
    fixed_cost_frame = CSVQA_FRAMES['file_1_view_0']
    trans_cost_frame = CSVQA_FRAMES['file_2_view_0']
    suppliers = []
    for (_, row) in fixed_cost_frame.iterrows():
        supplier_id = row['facility_id']
        if supplier_id not in suppliers:
            suppliers.append(supplier_id)
    customers = []
    for (_, row) in demand_frame.iterrows():
        customer_id = row['customer_id']
        if customer_id not in customers:
            customers.append(customer_id)
    demand = {}
    for (_, row) in demand_frame.iterrows():
        customer_id = row['customer_id']
        try:
            demand_units = float(row['demand_units'])
        except Exception:
            raise ValueError(f'Non-numeric demand_units for customer {customer_id}')
        if customer_id in demand:
            demand[customer_id] += demand_units
        else:
            demand[customer_id] = demand_units
    fixed_cost = {}
    for (_, row) in fixed_cost_frame.iterrows():
        supplier_id = row['facility_id']
        try:
            cost = float(row['fixed_opening_cost'])
        except Exception:
            raise ValueError(f'Non-numeric fixed_opening_cost for supplier {supplier_id}')
        if supplier_id in fixed_cost:
            fixed_cost[supplier_id] += cost
        else:
            fixed_cost[supplier_id] = cost
    cost = {}
    for (_, row) in trans_cost_frame.iterrows():
        supplier_id = row['facility_id']
        if supplier_id not in suppliers:
            continue
        if supplier_id not in cost:
            cost[supplier_id] = {}
        for customer_id in customers:
            colname = f'transportation_cost_to_{customer_id}'
            if colname not in trans_cost_frame.columns:
                raise ValueError(f'Missing transportation cost column {colname}')
            try:
                cij = float(row[colname])
            except Exception:
                raise ValueError(f'Non-numeric transportation cost for supplier {supplier_id} to customer {customer_id}')
            cost[supplier_id][customer_id] = cij
    for i in suppliers:
        if i not in cost:
            raise ValueError(f'Missing cost row for supplier {i}')
        for j in customers:
            if j not in cost[i]:
                raise ValueError(f'Missing cost for supplier {i} to customer {j}')
    M = sum((demand[j] for j in customers))
    M_i = {i: M for i in suppliers}
    m = gp.Model('Superstore_FLP')
    m.Params.MIPGap = 0.0001
    quantity_vars = m.addVars(suppliers, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
    open_vars = m.addVars(suppliers, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[i][j] * quantity_vars[i, j] for i in suppliers for j in customers)) + gp.quicksum((fixed_cost[i] * open_vars[i] for i in suppliers)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((quantity_vars[i, j] for i in suppliers)) == demand[j] for j in customers), name='')
    m.addConstrs((gp.quicksum((quantity_vars[i, j] for j in customers)) <= M_i[i] * open_vars[i] for i in suppliers), name='')
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')