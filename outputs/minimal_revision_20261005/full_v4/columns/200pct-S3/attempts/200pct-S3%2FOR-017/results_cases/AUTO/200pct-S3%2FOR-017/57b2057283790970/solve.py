CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'There are multiple suppliers responsible for delivering essential goods daily to customer groups located in '
          'different regions. Each supplier has a specific daily supply capacity, detailed in "supply_capacity.csv", '
          'while the daily demand of each customer group is recorded in "customer_demand.csv". The transportation cost '
          'per unit of goods from each supplier to each customer group is provided in "transportation_costs.csv". The '
          'objective is to determine the optimal transportation plan, specifying how goods should be allocated from '
          'each supplier to each customer group, ensuring that all demands are met without exceeding the supply '
          'capacity of any supplier, while minimizing the total transportation cost.',
 'relationships': [{'column_axis': {'id_column': 'customer_id', 'table_id': 'file_0_view_0'},
                    'column_id_mapping': {'transportation_cost_to_C1': 'C1',
                                          'transportation_cost_to_C10': 'C10',
                                          'transportation_cost_to_C2': 'C2',
                                          'transportation_cost_to_C3': 'C3',
                                          'transportation_cost_to_C4': 'C4',
                                          'transportation_cost_to_C5': 'C5',
                                          'transportation_cost_to_C6': 'C6',
                                          'transportation_cost_to_C7': 'C7',
                                          'transportation_cost_to_C8': 'C8',
                                          'transportation_cost_to_C9': 'C9'},
                    'matrix_table_id': 'file_2_view_0',
                    'row_axis': {'id_column': 'supplier_id', 'table_id': 'file_1_view_0'},
                    'row_id_column': 'supplier_id',
                    'type': 'matrix'}],
 'route': 'TP',
 'tables': [{'columns': ['customer_id', 'demand'],
             'file_index': 0,
             'file_name': 'customer_demand.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0, 'values': {'customer_id': 'C1', 'demand': '216'}},
                         {'source_row': 1, 'values': {'customer_id': 'C2', 'demand': '168'}},
                         {'source_row': 2, 'values': {'customer_id': 'C3', 'demand': '264'}},
                         {'source_row': 3, 'values': {'customer_id': 'C4', 'demand': '216'}},
                         {'source_row': 4, 'values': {'customer_id': 'C5', 'demand': '216'}},
                         {'source_row': 5, 'values': {'customer_id': 'C6', 'demand': '192'}},
                         {'source_row': 6, 'values': {'customer_id': 'C7', 'demand': '144'}},
                         {'source_row': 7, 'values': {'customer_id': 'C8', 'demand': '168'}},
                         {'source_row': 8, 'values': {'customer_id': 'C9', 'demand': '168'}},
                         {'source_row': 9, 'values': {'customer_id': 'C10', 'demand': '168'}}],
             'returned_rows': 10,
             'role': 'customer demand',
             'table_id': 'file_0_view_0'},
            {'columns': ['supplier_id', 'supply_capacity'],
             'file_index': 1,
             'file_name': 'supply_capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0, 'values': {'supplier_id': 'S1', 'supply_capacity': '288'}},
                         {'source_row': 1, 'values': {'supplier_id': 'S2', 'supply_capacity': '288'}},
                         {'source_row': 2, 'values': {'supplier_id': 'S3', 'supply_capacity': '264'}},
                         {'source_row': 3, 'values': {'supplier_id': 'S4', 'supply_capacity': '264'}},
                         {'source_row': 4, 'values': {'supplier_id': 'S5', 'supply_capacity': '216'}},
                         {'source_row': 5, 'values': {'supplier_id': 'S6', 'supply_capacity': '216'}},
                         {'source_row': 6, 'values': {'supplier_id': 'S7', 'supply_capacity': '168'}},
                         {'source_row': 7, 'values': {'supplier_id': 'S8', 'supply_capacity': '216'}},
                         {'source_row': 8, 'values': {'supplier_id': 'S9', 'supply_capacity': '240'}},
                         {'source_row': 9, 'values': {'supplier_id': 'S10', 'supply_capacity': '168'}}],
             'returned_rows': 10,
             'role': 'supplier capacity',
             'table_id': 'file_1_view_0'},
            {'columns': ['supplier_id',
                         'transportation_cost_to_C1',
                         'transportation_cost_to_C2',
                         'transportation_cost_to_C3',
                         'transportation_cost_to_C4',
                         'transportation_cost_to_C5',
                         'transportation_cost_to_C6',
                         'transportation_cost_to_C7',
                         'transportation_cost_to_C8',
                         'transportation_cost_to_C9',
                         'transportation_cost_to_C10'],
             'file_index': 2,
             'file_name': 'transportation_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0,
                          'values': {'supplier_id': 'S1',
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
                          'values': {'supplier_id': 'S2',
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
                          'values': {'supplier_id': 'S3',
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
                          'values': {'supplier_id': 'S4',
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
                          'values': {'supplier_id': 'S5',
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
                          'values': {'supplier_id': 'S6',
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
                          'values': {'supplier_id': 'S7',
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
                          'values': {'supplier_id': 'S8',
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
                          'values': {'supplier_id': 'S9',
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
                          'values': {'supplier_id': 'S10',
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
             'role': 'transportation cost matrix',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [{'column_ids_aligned': True,
                                   'column_mapping_basis': 'unique_complete_suffix',
                                   'expected_shape': [10, 10],
                                   'matrix_table_id': 'file_2_view_0',
                                   'row_ids_aligned': True,
                                   'row_mapping_basis': 'exact',
                                   'shape': [10, 10]}],
                'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    customer_demand_table = None
    supply_capacity_table = None
    transportation_costs_table = None
    for t in CSVQA_DATA['tables']:
        if t['table_id'] == 'file_0_view_0':
            customer_demand_table = t
        elif t['table_id'] == 'file_1_view_0':
            supply_capacity_table = t
        elif t['table_id'] == 'file_2_view_0':
            transportation_costs_table = t
    if customer_demand_table is None or supply_capacity_table is None or transportation_costs_table is None:
        raise RuntimeError('Missing required table(s) in CSVQA_DATA.')
    suppliers = [rec['values']['supplier_id'] for rec in supply_capacity_table['records']]
    customers = [rec['values']['customer_id'] for rec in customer_demand_table['records']]
    demand = {}
    for rec in customer_demand_table['records']:
        cid = rec['values']['customer_id']
        demand[cid] = float(rec['values']['demand'])
    supply_capacity = {}
    for rec in supply_capacity_table['records']:
        sid = rec['values']['supplier_id']
        supply_capacity[sid] = float(rec['values']['supply_capacity'])
    col_map = None
    for rel in CSVQA_DATA.get('relationships', []):
        if rel.get('matrix_table_id') == 'file_2_view_0':
            col_map = rel.get('column_id_mapping', None)
            break
    if col_map is None:
        raise RuntimeError('Missing column_id_mapping for transportation_costs.csv.')
    cost = {}
    for rec in transportation_costs_table['records']:
        sid = rec['values']['supplier_id']
        cost[sid] = {}
        for (col, cid) in col_map.items():
            if col not in rec['values']:
                raise RuntimeError(f'Missing cost column {col} for supplier {sid}.')
            cost[sid][cid] = float(rec['values'][col])
    for sid in suppliers:
        if sid not in cost:
            raise RuntimeError(f'Missing cost row for supplier {sid}.')
        for cid in customers:
            if cid not in cost[sid]:
                raise RuntimeError(f'Missing cost for supplier {sid}, customer {cid}.')
    for cid in customers:
        if cid not in demand:
            raise RuntimeError(f'Missing demand for customer {cid}.')
    for sid in suppliers:
        if sid not in supply_capacity:
            raise RuntimeError(f'Missing supply capacity for supplier {sid}.')
    m = gp.Model('Original_RAG_TP')
    m.Params.MIPGap = 0.0001
    x = m.addVars(suppliers, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[sid][cid] * x[sid, cid] for sid in suppliers for cid in customers)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[sid, cid] for sid in suppliers)) >= demand[cid] for cid in customers), name='')
    m.addConstrs((gp.quicksum((x[sid, cid] for cid in customers)) <= supply_capacity[sid] for sid in suppliers), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()