CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A retail chain, “GreenMart,” operates several warehouses that supply products to its various store '
          'locations. The daily demand for each store is provided in “customer_demand.csv,” while the daily supply '
          'capacity of each warehouse is detailed in “supply_capacity.csv.” The cost of transporting each unit of '
          'product from each warehouse to each store is recorded in “transportation_costs.csv.” The objective is to '
          'determine the optimal quantity of products to be shipped from each warehouse to each GreenMart store, '
          'ensuring that all store demands are met without exceeding the supply capacity of any warehouse, while '
          'minimizing the total transportation cost.',
 'relationships': [{'column_axis': {'id_column': 'customer', 'table_id': 'file_0_view_0'},
                    'matrix_table_id': 'file_2_view_0',
                    'row_axis': {'id_column': 'region', 'table_id': 'file_1_view_0'},
                    'row_id_column': 'Unnamed: 0',
                    'type': 'matrix'}],
 'route': 'TP',
 'tables': [{'columns': ['customer', 'demand'],
             'file_index': 0,
             'file_name': 'customer_demand.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0, 'values': {'customer': 'D1', 'demand': '428'}},
                         {'source_row': 1, 'values': {'customer': 'D2', 'demand': '217'}},
                         {'source_row': 2, 'values': {'customer': 'D3', 'demand': '214'}},
                         {'source_row': 3, 'values': {'customer': 'D4', 'demand': '380'}},
                         {'source_row': 4, 'values': {'customer': 'D5', 'demand': '254'}}],
             'returned_rows': 5,
             'role': 'store demand',
             'table_id': 'file_0_view_0'},
            {'columns': ['region', 'supply_capacity'],
             'file_index': 1,
             'file_name': 'supply_capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0, 'values': {'region': 'S1', 'supply_capacity': '428'}},
                         {'source_row': 1, 'values': {'region': 'S2', 'supply_capacity': '217'}},
                         {'source_row': 2, 'values': {'region': 'S3', 'supply_capacity': '214'}},
                         {'source_row': 3, 'values': {'region': 'S4', 'supply_capacity': '380'}},
                         {'source_row': 4, 'values': {'region': 'S5', 'supply_capacity': '254'}}],
             'returned_rows': 5,
             'role': 'warehouse supply capacity',
             'table_id': 'file_1_view_0'},
            {'columns': ['Unnamed: 0', 'D1', 'D2', 'D3', 'D4', 'D5'],
             'file_index': 2,
             'file_name': 'transportation_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0,
                          'values': {'D1': '269.3910588020795',
                                     'D2': '1.4537335390933939',
                                     'D3': '99.60345345756605',
                                     'D4': '26.64078166309837',
                                     'D5': '9.537688956880922',
                                     'Unnamed: 0': 'S1'}},
                         {'source_row': 1,
                          'values': {'D1': '9.291846876785183',
                                     'D2': '10.874778437070223',
                                     'D3': '144.52609291614627',
                                     'D4': '11.420133077898234',
                                     'D5': '153.1756819927813',
                                     'Unnamed: 0': 'S2'}},
                         {'source_row': 2,
                          'values': {'D1': '9.674584301671008',
                                     'D2': '2.6191650959687944',
                                     'D3': '100.8242249168735',
                                     'D4': '3.2121910887916876',
                                     'D5': '133.8493396124168',
                                     'Unnamed: 0': 'S3'}},
                         {'source_row': 3,
                          'values': {'D1': '270.57498480010247',
                                     'D2': '32.50253586',
                                     'D3': '4.6842098096469815',
                                     'D4': '1.5682269686546804',
                                     'D5': '9.58927599',
                                     'Unnamed: 0': 'S4'}},
                         {'source_row': 4,
                          'values': {'D1': '226.0331910675782',
                                     'D2': '8.669161980826471',
                                     'D3': '65.47681316968448',
                                     'D4': '9.068765258459958',
                                     'D5': '202.65015316425533',
                                     'Unnamed: 0': 'S5'}}],
             'returned_rows': 5,
             'role': 'transportation cost matrix',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [{'column_ids_aligned': True,
                                   'column_mapping_basis': 'exact',
                                   'expected_shape': [5, 5],
                                   'matrix_table_id': 'file_2_view_0',
                                   'row_ids_aligned': True,
                                   'row_mapping_basis': 'exact',
                                   'shape': [5, 5]}],
                'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    demand_table = [rec['values'] for rec in CSVQA_DATA['tables'][0]['records']]
    customers = [rec['customer'] for rec in demand_table]
    demand = {}
    for rec in demand_table:
        cust = rec['customer']
        try:
            demand[cust] = float(rec['demand'])
        except Exception:
            raise ValueError(f"Invalid demand value for customer {cust}: {rec['demand']}")
    supply_table = [rec['values'] for rec in CSVQA_DATA['tables'][1]['records']]
    warehouses = [rec['region'] for rec in supply_table]
    supply_capacity = {}
    for rec in supply_table:
        wh = rec['region']
        try:
            supply_capacity[wh] = float(rec['supply_capacity'])
        except Exception:
            raise ValueError(f"Invalid supply_capacity value for warehouse {wh}: {rec['supply_capacity']}")
    cost_table = [rec['values'] for rec in CSVQA_DATA['tables'][2]['records']]
    cost_warehouses = [rec['Unnamed: 0'] for rec in cost_table]
    cost_customers = [col for col in CSVQA_DATA['tables'][2]['columns'] if col != 'Unnamed: 0']
    if set(cost_warehouses) != set(warehouses):
        raise ValueError('Mismatch between warehouse IDs in supply_capacity and transportation_costs')
    if set(cost_customers) != set(customers):
        raise ValueError('Mismatch between customer IDs in customer_demand and transportation_costs')
    cost = {}
    for rec in cost_table:
        wh = rec['Unnamed: 0']
        cost[wh] = {}
        for cust in customers:
            try:
                cost[wh][cust] = float(rec[cust])
            except Exception:
                raise ValueError(f'Invalid cost value for warehouse {wh}, customer {cust}: {rec[cust]}')
    if set(demand.keys()) != set(customers):
        raise ValueError('Demand keys do not match customer list')
    if set(supply_capacity.keys()) != set(warehouses):
        raise ValueError('Supply capacity keys do not match warehouse list')
    for wh in warehouses:
        if wh not in cost or set(cost[wh].keys()) != set(customers):
            raise ValueError(f'Cost data missing for warehouse {wh}')
    m = gp.Model('GreenMart_TP')
    m.Params.MIPGap = 0.0001
    x = m.addVars(warehouses, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[wh][cust] * x[wh, cust] for wh in warehouses for cust in customers)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[wh, cust] for wh in warehouses)) >= demand[cust] for cust in customers), name='')
    m.addConstrs((gp.quicksum((x[wh, cust] for cust in customers)) <= supply_capacity[wh] for wh in warehouses), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()