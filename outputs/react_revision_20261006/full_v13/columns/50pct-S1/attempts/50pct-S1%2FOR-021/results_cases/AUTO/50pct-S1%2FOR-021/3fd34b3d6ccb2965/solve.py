CSVQA_DATA = {'ignored_file_indices': [],
 'query': '‚ÄúBrewCo,‚Äù a beverage manufacturer, operates multiple production facilities that distribute drinks to '
          'various retail locations. The daily demand for each retail outlet is specified in '
          '‚Äúcustomer_demand.csv,‚Äù while the production capacity of each plant is outlined in '
          '‚Äúsupply_capacity.csv.‚Äù The transportation cost per unit of beverages from each plant to each outlet is '
          'recorded in ‚Äútransportation_costs.csv.‚Äù The goal is to determine the optimal quantity of beverages to '
          'be shipped from each production plant to each retail outlet, ensuring all outlet demands are met without '
          'surpassing any plant‚Äôs production capacity, while minimizing the total transportation cost.',
 'relationships': [],
 'route': 'TP',
 'tables': [{'columns': ['customer_id', 'archive_revision_number', 'demand'],
             'file_index': 0,
             'file_name': 'customer_demand.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 4,
             'records': [{'source_row': 0,
                          'values': {'archive_revision_number': '6', 'customer_id': 'C1', 'demand': '94'}},
                         {'source_row': 1,
                          'values': {'archive_revision_number': '2', 'customer_id': 'C2', 'demand': '39'}},
                         {'source_row': 2,
                          'values': {'archive_revision_number': '6', 'customer_id': 'C3', 'demand': '65'}},
                         {'source_row': 3,
                          'values': {'archive_revision_number': '5', 'customer_id': 'C4', 'demand': '435'}}],
             'returned_rows': 4,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['supplier_id', 'archive_revision_number', 'supply_capacity'],
             'file_index': 1,
             'file_name': 'supply_capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 4,
             'records': [{'source_row': 0,
                          'values': {'archive_revision_number': '2', 'supplier_id': 'S1', 'supply_capacity': '2531'}},
                         {'source_row': 1,
                          'values': {'archive_revision_number': '3', 'supplier_id': 'S2', 'supply_capacity': '20'}},
                         {'source_row': 2,
                          'values': {'archive_revision_number': '6', 'supplier_id': 'S3', 'supply_capacity': '210'}},
                         {'source_row': 3,
                          'values': {'archive_revision_number': '4', 'supplier_id': 'S4', 'supply_capacity': '241'}}],
             'returned_rows': 4,
             'role': 'file_1',
             'table_id': 'file_1_view_0'},
            {'columns': ['supplier_id',
                         'transportation_cost_to_C1',
                         'record_display_theme',
                         'transportation_cost_to_C2',
                         'transportation_cost_to_C3',
                         'document_page_count',
                         'archive_revision_number',
                         'transportation_cost_to_C4'],
             'file_index': 2,
             'file_name': 'transportation_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 4,
             'records': [{'source_row': 0,
                          'values': {'archive_revision_number': '4',
                                     'document_page_count': '2',
                                     'record_display_theme': 'Azure',
                                     'supplier_id': 'S1',
                                     'transportation_cost_to_C1': '543.756480860856',
                                     'transportation_cost_to_C2': '23.685276141764653',
                                     'transportation_cost_to_C3': '23.676386730773032',
                                     'transportation_cost_to_C4': '447.75143678673766'}},
                         {'source_row': 1,
                          'values': {'archive_revision_number': '4',
                                     'document_page_count': '8',
                                     'record_display_theme': 'Amber',
                                     'supplier_id': 'S2',
                                     'transportation_cost_to_C1': '883.9151090405642',
                                     'transportation_cost_to_C2': '0.04977684765576961',
                                     'transportation_cost_to_C3': '0.0350986687216299',
                                     'transportation_cost_to_C4': '44.45588531711622'}},
                         {'source_row': 2,
                          'values': {'archive_revision_number': '1',
                                     'document_page_count': '4',
                                     'record_display_theme': 'Amber',
                                     'supplier_id': 'S3',
                                     'transportation_cost_to_C1': '537.3456896658107',
                                     'transportation_cost_to_C2': '23.769274659075112',
                                     'transportation_cost_to_C3': '498.95659249465467',
                                     'transportation_cost_to_C4': '440.60737890439776'}},
                         {'source_row': 3,
                          'values': {'archive_revision_number': '3',
                                     'document_page_count': '12',
                                     'record_display_theme': 'Olive',
                                     'supplier_id': 'S4',
                                     'transportation_cost_to_C1': '1791.493192397229',
                                     'transportation_cost_to_C2': '68.21633865655126',
                                     'transportation_cost_to_C3': '1432.4837339656747',
                                     'transportation_cost_to_C4': '1527.7635425462734'}}],
             'returned_rows': 4,
             'role': 'file_2',
             'table_id': 'file_2_view_0'}],
 'validation': {'fallback_reason': "Matrix validation failed: {'matrix_table_id': 'file_2_view_0', 'shape': [4, 5], "
                                   "'expected_shape': [4, 4], 'row_ids_aligned': True, 'column_ids_aligned': False, "
                                   "'row_mapping_basis': 'exact', 'column_mapping_basis': 'unresolved'}",
                'planner_errors': ["Matrix validation failed: {'matrix_table_id': 'file_2_view_0', 'shape': [4, 5], "
                                   "'expected_shape': [4, 4], 'row_ids_aligned': True, 'column_ids_aligned': False, "
                                   "'row_mapping_basis': 'exact', 'column_mapping_basis': 'unresolved'}"],
                'status': 'FALLBACK_FULL_DATA'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem(CSVQA_FRAMES):
    supply_frame = CSVQA_FRAMES['file_1_view_0']
    demand_frame = CSVQA_FRAMES['file_0_view_0']
    cost_frame = CSVQA_FRAMES['file_2_view_0']
    plants = []
    for (_, row) in supply_frame.iterrows():
        plant_id = row['supplier_id']
        if plant_id not in plants:
            plants.append(plant_id)
    outlets = []
    for (_, row) in demand_frame.iterrows():
        outlet_id = row['customer_id']
        if outlet_id not in outlets:
            outlets.append(outlet_id)
    demand = {}
    for (_, row) in demand_frame.iterrows():
        outlet_id = row['customer_id']
        demand[outlet_id] = float(row['demand'])
    supply_capacity = {}
    for (_, row) in supply_frame.iterrows():
        plant_id = row['supplier_id']
        supply_capacity[plant_id] = float(row['supply_capacity'])
    cost = {}
    for (_, row) in cost_frame.iterrows():
        plant_id = row['supplier_id']
        cost[plant_id] = {}
        for outlet_id in outlets:
            col = f'transportation_cost_to_{outlet_id}'
            cost[plant_id][outlet_id] = float(row[col])
    for plant_id in plants:
        if plant_id not in cost or plant_id not in supply_capacity:
            raise ValueError(f'Missing cost or supply_capacity for plant {plant_id}')
        for outlet_id in outlets:
            if outlet_id not in cost[plant_id]:
                raise ValueError(f'Missing cost for plant {plant_id}, outlet {outlet_id}')
    for outlet_id in outlets:
        if outlet_id not in demand:
            raise ValueError(f'Missing demand for outlet {outlet_id}')
    m = gp.Model('BrewCo_Transportation')
    m.Params.MIPGap = 0.0001
    quantity_vars = m.addVars(plants, outlets, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[i][j] * quantity_vars[i, j] for i in plants for j in outlets)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((quantity_vars[i, j] for i in plants)) >= demand[j] for j in outlets), name='')
    m.addConstrs((gp.quicksum((quantity_vars[i, j] for j in outlets)) <= supply_capacity[i] for i in plants), name='')
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')