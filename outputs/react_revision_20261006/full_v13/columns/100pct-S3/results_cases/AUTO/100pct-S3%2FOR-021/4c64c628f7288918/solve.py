CSVQA_DATA = {'ignored_file_indices': [],
 'query': '‚ÄúBrewCo,‚Äù a beverage manufacturer, operates multiple production facilities that distribute drinks to '
          'various retail locations. The daily demand for each retail outlet is specified in '
          '‚Äúcustomer_demand.csv,‚Äù while the production capacity of each plant is outlined in '
          '‚Äúsupply_capacity.csv.‚Äù The transportation cost per unit of beverages from each plant to each outlet is '
          'recorded in ‚Äútransportation_costs.csv.‚Äù The goal is to determine the optimal quantity of beverages to '
          'be shipped from each production plant to each retail outlet, ensuring all outlet demands are met without '
          'surpassing any plant‚Äôs production capacity, while minimizing the total transportation cost.',
 'relationships': [{'column_axis': {'id_column': 'customer_id', 'table_id': 'file_0_view_0'},
                    'column_id_mapping': {'transportation_cost_to_C1': 'C1',
                                          'transportation_cost_to_C2': 'C2',
                                          'transportation_cost_to_C3': 'C3',
                                          'transportation_cost_to_C4': 'C4'},
                    'matrix_table_id': 'file_2_view_0',
                    'row_axis': {'id_column': 'supplier_id', 'table_id': 'file_1_view_0'},
                    'row_id_column': 'supplier_id',
                    'type': 'matrix'}],
 'route': 'TP',
 'tables': [{'columns': ['customer_id', 'demand'],
             'file_index': 0,
             'file_name': 'customer_demand.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 4,
             'records': [{'source_row': 0, 'values': {'customer_id': 'C1', 'demand': '94'}},
                         {'source_row': 1, 'values': {'customer_id': 'C2', 'demand': '39'}},
                         {'source_row': 2, 'values': {'customer_id': 'C3', 'demand': '65'}},
                         {'source_row': 3, 'values': {'customer_id': 'C4', 'demand': '435'}}],
             'returned_rows': 4,
             'role': 'retail outlet demand',
             'table_id': 'file_0_view_0'},
            {'columns': ['supplier_id', 'supply_capacity'],
             'file_index': 1,
             'file_name': 'supply_capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 4,
             'records': [{'source_row': 0, 'values': {'supplier_id': 'S1', 'supply_capacity': '2531'}},
                         {'source_row': 1, 'values': {'supplier_id': 'S2', 'supply_capacity': '20'}},
                         {'source_row': 2, 'values': {'supplier_id': 'S3', 'supply_capacity': '210'}},
                         {'source_row': 3, 'values': {'supplier_id': 'S4', 'supply_capacity': '241'}}],
             'returned_rows': 4,
             'role': 'production plant capacity',
             'table_id': 'file_1_view_0'},
            {'columns': ['supplier_id',
                         'transportation_cost_to_C1',
                         'transportation_cost_to_C2',
                         'transportation_cost_to_C3',
                         'transportation_cost_to_C4'],
             'file_index': 2,
             'file_name': 'transportation_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 4,
             'records': [{'source_row': 0,
                          'values': {'supplier_id': 'S1',
                                     'transportation_cost_to_C1': '543.756480860856',
                                     'transportation_cost_to_C2': '23.685276141764653',
                                     'transportation_cost_to_C3': '23.676386730773032',
                                     'transportation_cost_to_C4': '447.75143678673766'}},
                         {'source_row': 1,
                          'values': {'supplier_id': 'S2',
                                     'transportation_cost_to_C1': '883.9151090405642',
                                     'transportation_cost_to_C2': '0.04977684765576961',
                                     'transportation_cost_to_C3': '0.0350986687216299',
                                     'transportation_cost_to_C4': '44.45588531711622'}},
                         {'source_row': 2,
                          'values': {'supplier_id': 'S3',
                                     'transportation_cost_to_C1': '537.3456896658107',
                                     'transportation_cost_to_C2': '23.769274659075112',
                                     'transportation_cost_to_C3': '498.95659249465467',
                                     'transportation_cost_to_C4': '440.60737890439776'}},
                         {'source_row': 3,
                          'values': {'supplier_id': 'S4',
                                     'transportation_cost_to_C1': '1791.493192397229',
                                     'transportation_cost_to_C2': '68.21633865655126',
                                     'transportation_cost_to_C3': '1432.4837339656747',
                                     'transportation_cost_to_C4': '1527.7635425462734'}}],
             'returned_rows': 4,
             'role': 'plant-to-outlet transportation cost matrix',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [{'column_ids_aligned': True,
                                   'column_mapping_basis': 'unique_complete_suffix',
                                   'expected_shape': [4, 4],
                                   'matrix_table_id': 'file_2_view_0',
                                   'row_ids_aligned': True,
                                   'row_mapping_basis': 'exact',
                                   'shape': [4, 4]}],
                'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem(CSVQA_FRAMES):
    plants_frame = CSVQA_FRAMES['file_1_view_0']
    outlets_frame = CSVQA_FRAMES['file_0_view_0']
    cost_frame = CSVQA_FRAMES['file_2_view_0']
    plants = []
    plant_supply = {}
    for (_, row) in plants_frame.iterrows():
        pid = row['supplier_id']
        plants.append(pid)
        try:
            plant_supply[pid] = float(row['supply_capacity'])
        except Exception:
            raise ValueError(f"Invalid supply_capacity for plant {pid}: {row['supply_capacity']}")
    outlets = []
    outlet_demand = {}
    for (_, row) in outlets_frame.iterrows():
        cid = row['customer_id']
        outlets.append(cid)
        try:
            outlet_demand[cid] = float(row['demand'])
        except Exception:
            raise ValueError(f"Invalid demand for outlet {cid}: {row['demand']}")
    column_id_mapping = {'transportation_cost_to_C1': 'C1', 'transportation_cost_to_C2': 'C2', 'transportation_cost_to_C3': 'C3', 'transportation_cost_to_C4': 'C4'}
    cost = {}
    for (_, row) in cost_frame.iterrows():
        pid = row['supplier_id']
        cost[pid] = {}
        for (col, cid) in column_id_mapping.items():
            try:
                cost[pid][cid] = float(row[col])
            except Exception:
                raise ValueError(f'Invalid cost for plant {pid}, outlet {cid}: {row[col]}')
    if set(plants) != set(cost.keys()):
        raise ValueError('Mismatch between plants in supply_capacity and transportation_costs.')
    for pid in plants:
        if set(outlets) != set(cost[pid].keys()):
            raise ValueError(f'Mismatch between outlets in cost row for plant {pid}.')
    m = gp.Model('BrewCo_Transportation')
    m.Params.MIPGap = 0.0001
    quantity_vars = m.addVars(plants, outlets, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[i][j] * quantity_vars[i, j] for i in plants for j in outlets)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((quantity_vars[i, j] for i in plants)) >= outlet_demand[j] for j in outlets), name='')
    m.addConstrs((gp.quicksum((quantity_vars[i, j] for j in outlets)) <= plant_supply[i] for i in plants), name='')
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)