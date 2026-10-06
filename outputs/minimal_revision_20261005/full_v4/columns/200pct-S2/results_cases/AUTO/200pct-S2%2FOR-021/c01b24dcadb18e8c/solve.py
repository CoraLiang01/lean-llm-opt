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
             'role': 'customer demand',
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
             'role': 'supply capacity',
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
             'role': 'transportation cost matrix',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [{'column_ids_aligned': True,
                                   'column_mapping_basis': 'unique_complete_suffix',
                                   'expected_shape': [4, 4],
                                   'matrix_table_id': 'file_2_view_0',
                                   'row_ids_aligned': True,
                                   'row_mapping_basis': 'exact',
                                   'shape': [4, 4]}],
                'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    customer_table = 'file_0_view_0'
    supplier_table = 'file_1_view_0'
    cost_table = 'file_2_view_0'
    customer_records = [rec['values'] for rec in CSVQA_DATA['tables'] if rec['table_id'] == customer_table][0:][0]['records'] if False else None
    for t in CSVQA_DATA['tables']:
        if t['table_id'] == customer_table:
            customer_records = t['records']
            break
    customers = []
    demand = {}
    for rec in customer_records:
        cid = rec['values']['customer_id']
        customers.append(cid)
        demand[cid] = float(rec['values']['demand'])
    supplier_records = [rec['values'] for rec in CSVQA_DATA['tables'] if rec['table_id'] == supplier_table][0:][0]['records'] if False else None
    for t in CSVQA_DATA['tables']:
        if t['table_id'] == supplier_table:
            supplier_records = t['records']
            break
    suppliers = []
    supply_capacity = {}
    for rec in supplier_records:
        sid = rec['values']['supplier_id']
        suppliers.append(sid)
        supply_capacity[sid] = float(rec['values']['supply_capacity'])
    for rel in CSVQA_DATA['relationships']:
        if rel['matrix_table_id'] == cost_table:
            row_id_column = rel['row_axis']['id_column']
            row_table_id = rel['row_axis']['table_id']
            col_id_mapping = rel.get('column_id_mapping', {})
            break
    cost_records = None
    for t in CSVQA_DATA['tables']:
        if t['table_id'] == cost_table:
            cost_records = t['records']
            break
    cost = {}
    for rec in cost_records:
        sid = rec['values']['supplier_id']
        cost[sid] = {}
        for (col_csv, cid) in col_id_mapping.items():
            cost[sid][cid] = float(rec['values'][col_csv])
    if set(suppliers) != set(cost.keys()):
        raise ValueError('Mismatch between suppliers and cost matrix rows')
    for sid in suppliers:
        if set(customers) != set(cost[sid].keys()):
            raise ValueError(f'Mismatch between customers and cost matrix columns for supplier {sid}')
    m = gp.Model('BrewCo_Transportation')
    x = m.addVars(suppliers, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in suppliers for j in customers)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in suppliers)) >= demand[j] for j in customers), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in customers)) <= supply_capacity[i] for i in suppliers), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()