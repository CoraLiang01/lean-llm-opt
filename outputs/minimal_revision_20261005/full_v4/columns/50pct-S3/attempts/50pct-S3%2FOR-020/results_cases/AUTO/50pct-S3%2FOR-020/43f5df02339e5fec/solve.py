CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A retail chain, ‚ÄúGreenMart,‚Äù operates several warehouses that supply products to its various store '
          'locations. The daily demand for each store is provided in ‚Äúcustomer_demand.csv,‚Äù while the daily supply '
          'capacity of each warehouse is detailed in ‚Äúsupply_capacity.csv.‚Äù The cost of transporting each unit of '
          'product from each warehouse to each store is recorded in ‚Äútransportation_costs.csv.‚Äù The objective is '
          'to determine the optimal quantity of products to be shipped from each warehouse to each GreenMart store, '
          'ensuring that all store demands are met without exceeding the supply capacity of any warehouse, while '
          'minimizing the total transportation cost.',
 'relationships': [{'column_axis': {'id_column': 'customer_id', 'table_id': 'file_0_view_0'},
                    'column_id_mapping': {'transportation_cost_to_D1': 'D1',
                                          'transportation_cost_to_D2': 'D2',
                                          'transportation_cost_to_D3': 'D3',
                                          'transportation_cost_to_D4': 'D4',
                                          'transportation_cost_to_D5': 'D5'},
                    'matrix_table_id': 'file_2_view_0',
                    'row_axis': {'id_column': 'supplier_id', 'table_id': 'file_1_view_0'},
                    'row_id_column': 'supplier_id',
                    'type': 'matrix'}],
 'route': 'TP',
 'tables': [{'columns': ['customer_id', 'demand_units'],
             'file_index': 0,
             'file_name': 'customer_demand.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0, 'values': {'customer_id': 'D1', 'demand_units': '428'}},
                         {'source_row': 1, 'values': {'customer_id': 'D2', 'demand_units': '217'}},
                         {'source_row': 2, 'values': {'customer_id': 'D3', 'demand_units': '214'}},
                         {'source_row': 3, 'values': {'customer_id': 'D4', 'demand_units': '380'}},
                         {'source_row': 4, 'values': {'customer_id': 'D5', 'demand_units': '254'}}],
             'returned_rows': 5,
             'role': 'store demand',
             'table_id': 'file_0_view_0'},
            {'columns': ['supplier_id', 'supply_capacity_units'],
             'file_index': 1,
             'file_name': 'supply_capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0, 'values': {'supplier_id': 'S1', 'supply_capacity_units': '428'}},
                         {'source_row': 1, 'values': {'supplier_id': 'S2', 'supply_capacity_units': '217'}},
                         {'source_row': 2, 'values': {'supplier_id': 'S3', 'supply_capacity_units': '214'}},
                         {'source_row': 3, 'values': {'supplier_id': 'S4', 'supply_capacity_units': '380'}},
                         {'source_row': 4, 'values': {'supplier_id': 'S5', 'supply_capacity_units': '254'}}],
             'returned_rows': 5,
             'role': 'warehouse supply capacity',
             'table_id': 'file_1_view_0'},
            {'columns': ['supplier_id',
                         'transportation_cost_to_D1',
                         'transportation_cost_to_D2',
                         'transportation_cost_to_D3',
                         'transportation_cost_to_D4',
                         'transportation_cost_to_D5'],
             'file_index': 2,
             'file_name': 'transportation_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0,
                          'values': {'supplier_id': 'S1',
                                     'transportation_cost_to_D1': '269.3910588020795',
                                     'transportation_cost_to_D2': '1.453733539093394',
                                     'transportation_cost_to_D3': '99.60345345756603',
                                     'transportation_cost_to_D4': '26.64078166309837',
                                     'transportation_cost_to_D5': '9.537688956880922'}},
                         {'source_row': 1,
                          'values': {'supplier_id': 'S2',
                                     'transportation_cost_to_D1': '9.291846876785185',
                                     'transportation_cost_to_D2': '10.874778437070225',
                                     'transportation_cost_to_D3': '144.52609291614627',
                                     'transportation_cost_to_D4': '11.420133077898234',
                                     'transportation_cost_to_D5': '153.1756819927813'}},
                         {'source_row': 2,
                          'values': {'supplier_id': 'S3',
                                     'transportation_cost_to_D1': '9.674584301671008',
                                     'transportation_cost_to_D2': '2.6191650959687944',
                                     'transportation_cost_to_D3': '100.8242249168735',
                                     'transportation_cost_to_D4': '3.212191088791688',
                                     'transportation_cost_to_D5': '133.8493396124168'}},
                         {'source_row': 3,
                          'values': {'supplier_id': 'S4',
                                     'transportation_cost_to_D1': '270.57498480010247',
                                     'transportation_cost_to_D2': '32.50253586',
                                     'transportation_cost_to_D3': '4.6842098096469815',
                                     'transportation_cost_to_D4': '1.5682269686546804',
                                     'transportation_cost_to_D5': '9.58927599'}},
                         {'source_row': 4,
                          'values': {'supplier_id': 'S5',
                                     'transportation_cost_to_D1': '226.0331910675782',
                                     'transportation_cost_to_D2': '8.669161980826471',
                                     'transportation_cost_to_D3': '65.47681316968448',
                                     'transportation_cost_to_D4': '9.068765258459958',
                                     'transportation_cost_to_D5': '202.65015316425533'}}],
             'returned_rows': 5,
             'role': 'transportation cost matrix',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [{'column_ids_aligned': True,
                                   'column_mapping_basis': 'unique_complete_suffix',
                                   'expected_shape': [5, 5],
                                   'matrix_table_id': 'file_2_view_0',
                                   'row_ids_aligned': True,
                                   'row_mapping_basis': 'exact',
                                   'shape': [5, 5]}],
                'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    warehouses = []
    for rec in CSVQA_DATA['tables']:
        if rec['table_id'] == 'file_1_view_0':
            warehouses = [r['values']['supplier_id'] for r in rec['records']]
            break
    stores = []
    for rec in CSVQA_DATA['tables']:
        if rec['table_id'] == 'file_0_view_0':
            stores = [r['values']['customer_id'] for r in rec['records']]
            break
    demand = {}
    for rec in CSVQA_DATA['tables']:
        if rec['table_id'] == 'file_0_view_0':
            for r in rec['records']:
                cid = r['values']['customer_id']
                val = r['values']['demand_units']
                demand[cid] = float(val)
            break
    supply_capacity = {}
    for rec in CSVQA_DATA['tables']:
        if rec['table_id'] == 'file_1_view_0':
            for r in rec['records']:
                sid = r['values']['supplier_id']
                val = r['values']['supply_capacity_units']
                supply_capacity[sid] = float(val)
            break
    cost = {}
    for rec in CSVQA_DATA['tables']:
        if rec['table_id'] == 'file_2_view_0':
            for r in rec['records']:
                sid = r['values']['supplier_id']
                cost[sid] = {}
                for j in stores:
                    col = f'transportation_cost_to_{j}'
                    if col not in r['values']:
                        raise ValueError(f'Missing cost for {sid},{j}')
                    cost[sid][j] = float(r['values'][col])
            break
    for j in stores:
        if j not in demand:
            raise ValueError(f'Missing demand for store {j}')
    for i in warehouses:
        if i not in supply_capacity:
            raise ValueError(f'Missing supply for warehouse {i}')
        if i not in cost:
            raise ValueError(f'Missing cost row for warehouse {i}')
        for j in stores:
            if j not in cost[i]:
                raise ValueError(f'Missing cost for {i},{j}')
    m = gp.Model('GreenMart_Transportation')
    m.Params.MIPGap = 0.0001
    x = m.addVars(warehouses, stores, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in warehouses for j in stores)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in warehouses)) >= demand[j] for j in stores), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in stores)) <= supply_capacity[i] for i in warehouses), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()