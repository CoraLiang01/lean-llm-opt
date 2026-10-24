CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A logistics company manages multiple warehouses responsible for distributing products to various retail '
          'stores. The daily demand for each store is provided in “customer_demand.csv,” while the daily supply '
          'capacity of each warehouse is detailed in “supply_capacity.csv.” The cost of transporting each unit of '
          'product from each warehouse to each store is recorded in “transportation_costs.csv.” The objective is to '
          'determine the optimal quantity of products to be shipped from each warehouse to each store, ensuring that '
          'all store demands are met without exceeding the supply capacity of any warehouse, while minimizing the '
          'total transportation cost.',
 'relationships': [{'column_axis': {'id_column': 'customer', 'table_id': 'file_0_view_0'},
                    'matrix_table_id': 'file_2_view_0',
                    'row_axis': {'id_column': 'Unnamed: 0', 'table_id': 'file_1_view_0'},
                    'row_id_column': 'Unnamed: 0',
                    'type': 'matrix'}],
 'route': 'TP',
 'tables': [{'columns': ['customer', 'demand'],
             'file_index': 0,
             'file_name': 'customer_demand.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0, 'values': {'customer': 'C1', 'demand': '45'}},
                         {'source_row': 1, 'values': {'customer': 'C2', 'demand': '23'}},
                         {'source_row': 2, 'values': {'customer': 'C3', 'demand': '94'}},
                         {'source_row': 3, 'values': {'customer': 'C4', 'demand': '92'}},
                         {'source_row': 4, 'values': {'customer': 'C5', 'demand': '57'}},
                         {'source_row': 5, 'values': {'customer': 'C6', 'demand': '52'}},
                         {'source_row': 6, 'values': {'customer': 'C7', 'demand': '23'}},
                         {'source_row': 7, 'values': {'customer': 'C8', 'demand': '99'}},
                         {'source_row': 8, 'values': {'customer': 'C9', 'demand': '99'}},
                         {'source_row': 9, 'values': {'customer': 'C10', 'demand': '77'}}],
             'returned_rows': 10,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'},
            {'columns': ['Unnamed: 0', 'supply_capacity'],
             'file_index': 1,
             'file_name': 'supply_capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0, 'values': {'Unnamed: 0': 'S1', 'supply_capacity': '127'}},
                         {'source_row': 1, 'values': {'Unnamed: 0': 'S2', 'supply_capacity': '236'}},
                         {'source_row': 2, 'values': {'Unnamed: 0': 'S3', 'supply_capacity': '168'}},
                         {'source_row': 3, 'values': {'Unnamed: 0': 'S4', 'supply_capacity': '115'}},
                         {'source_row': 4, 'values': {'Unnamed: 0': 'S5', 'supply_capacity': '280'}},
                         {'source_row': 5, 'values': {'Unnamed: 0': 'S6', 'supply_capacity': '179'}},
                         {'source_row': 6, 'values': {'Unnamed: 0': 'S7', 'supply_capacity': '135'}},
                         {'source_row': 7, 'values': {'Unnamed: 0': 'S8', 'supply_capacity': '263'}},
                         {'source_row': 8, 'values': {'Unnamed: 0': 'S9', 'supply_capacity': '283'}},
                         {'source_row': 9, 'values': {'Unnamed: 0': 'S10', 'supply_capacity': '476'}}],
             'returned_rows': 10,
             'role': 'descriptive non-unique role',
             'table_id': 'file_1_view_0'},
            {'columns': ['Unnamed: 0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10'],
             'file_index': 2,
             'file_name': 'transportation_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0,
                          'values': {'C1': '2077.058672521021',
                                     'C10': '0.0',
                                     'C2': '0.0',
                                     'C3': '54.33526480458508',
                                     'C4': '0.0',
                                     'C5': '0.0',
                                     'C6': '36.17284629162332',
                                     'C7': '0.0',
                                     'C8': '0.0',
                                     'C9': '169.33026926588778',
                                     'Unnamed: 0': 'S1'}},
                         {'source_row': 1,
                          'values': {'C1': '2077.058672521021',
                                     'C10': '0.0',
                                     'C2': '0.0',
                                     'C3': '1141.0405608962865',
                                     'C4': '0.0',
                                     'C5': '0.0',
                                     'C6': '651.1112332492198',
                                     'C7': '0.0',
                                     'C8': '0.0',
                                     'C9': '8.063346155518467',
                                     'Unnamed: 0': 'S2'}},
                         {'source_row': 2,
                          'values': {'C1': '79.9210295982608',
                                     'C10': '474.24509131006675',
                                     'C2': '474.24509131006675',
                                     'C3': '1477.0676289106607',
                                     'C4': '22.583099586193658',
                                     'C5': '474.24509131006675',
                                     'C6': '41.106596962251096',
                                     'C7': '474.24509131006675',
                                     'C8': '474.24509131006675',
                                     'C9': '624.162539502301',
                                     'Unnamed: 0': 'S3'}},
                         {'source_row': 3,
                          'values': {'C1': '1659.336929105112',
                                     'C10': '1029.6974643797064',
                                     'C2': '57.20541468776147',
                                     'C3': '186.1519048103841',
                                     'C4': '1201.3137084429907',
                                     'C5': '1029.6974643797064',
                                     'C6': '41.82210594495074',
                                     'C7': '57.20541468776147',
                                     'C8': '1201.3137084429907',
                                     'C9': '884.5633870657458',
                                     'Unnamed: 0': 'S4'}},
                         {'source_row': 4,
                          'values': {'C1': '1297.2567040858307',
                                     'C10': '1399.7932436376784',
                                     'C2': '77.76629131320436',
                                     'C3': '24.26760227579214',
                                     'C4': '1399.7932436376784',
                                     'C5': '77.76629131320436',
                                     'C6': '53.91161728496604',
                                     'C7': '1399.7932436376784',
                                     'C8': '77.76629131320436',
                                     'C9': '1255.115148013589',
                                     'Unnamed: 0': 'S5'}},
                         {'source_row': 5,
                          'values': {'C1': '1998.9090658724567',
                                     'C10': '1149.53596749779',
                                     'C2': '985.3165435695341',
                                     'C3': '2.8541686885814643',
                                     'C4': '1149.53596749779',
                                     'C5': '985.3165435695341',
                                     'C6': '730.6923647662475',
                                     'C7': '54.73980797608523',
                                     'C8': '985.3165435695341',
                                     'C9': '46.803102206463265',
                                     'Unnamed: 0': 'S6'}},
                         {'source_row': 6,
                          'values': {'C1': '1780.3360050180179',
                                     'C10': '0.0',
                                     'C2': '0.0',
                                     'C3': '1141.0405608962865',
                                     'C4': '0.0',
                                     'C5': '0.0',
                                     'C6': '36.17284629162332',
                                     'C7': '0.0',
                                     'C8': '0.0',
                                     'C9': '8.063346155518467',
                                     'Unnamed: 0': 'S7'}},
                         {'source_row': 7,
                          'values': {'C1': '75.40935896233042',
                                     'C10': '1338.1987290721909',
                                     'C2': '1338.1987290721909',
                                     'C3': '21.391345987195333',
                                     'C4': '74.34437383734394',
                                     'C5': '74.34437383734394',
                                     'C6': '937.3506239061463',
                                     'C7': '1338.1987290721909',
                                     'C8': '1338.1987290721909',
                                     'C9': '1392.1186581383768',
                                     'Unnamed: 0': 'S8'}},
                         {'source_row': 8,
                          'values': {'C1': '98.90755583433433',
                                     'C10': '0.0',
                                     'C2': '0.0',
                                     'C3': '978.0347664825314',
                                     'C4': '0.0',
                                     'C5': '0.0',
                                     'C6': '651.1112332492198',
                                     'C7': '0.0',
                                     'C8': '0.0',
                                     'C9': '169.33026926588778',
                                     'Unnamed: 0': 'S9'}},
                         {'source_row': 9,
                          'values': {'C1': '2077.058672521021',
                                     'C10': '0.0',
                                     'C2': '0.0',
                                     'C3': '54.33526480458508',
                                     'C4': '0.0',
                                     'C5': '0.0',
                                     'C6': '36.17284629162332',
                                     'C7': '0.0',
                                     'C8': '0.0',
                                     'C9': '145.1402307993324',
                                     'Unnamed: 0': 'S10'}}],
             'returned_rows': 10,
             'role': 'descriptive non-unique role',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [{'column_ids_aligned': True,
                                   'column_mapping_basis': 'exact',
                                   'expected_shape': [10, 10],
                                   'matrix_table_id': 'file_2_view_0',
                                   'row_ids_aligned': True,
                                   'row_mapping_basis': 'exact',
                                   'shape': [10, 10]}],
                'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']

    def get_table(table_id):
        for t in tables:
            if t['table_id'] == table_id:
                return t
        raise ValueError(f'Table {table_id} not found')
    demand_table = get_table('file_0_view_0')
    demand = {}
    for rec in demand_table['records']:
        vals = rec['values']
        cust = vals['customer']
        demand[cust] = float(vals['demand'])
    capacity_table = get_table('file_1_view_0')
    capacity = {}
    for rec in capacity_table['records']:
        vals = rec['values']
        wh = vals['Unnamed: 0']
        capacity[wh] = float(vals['supply_capacity'])
    cost_table = get_table('file_2_view_0')
    cost = {}
    cols = cost_table['columns']
    cust_cols = cols[1:]
    for rec in cost_table['records']:
        vals = rec['values']
        wh = vals['Unnamed: 0']
        cost[wh] = {}
        for c in cust_cols:
            cost[wh][c] = float(vals[c])
    W = list(capacity.keys())
    C = list(demand.keys())
    if not set(W).issubset(capacity.keys()):
        missing = set(W) - set(capacity.keys())
        raise ValueError(f'Missing capacity for warehouses: {missing}')
    if not set(C).issubset(demand.keys()):
        missing = set(C) - set(demand.keys())
        raise ValueError(f'Missing demand for customers: {missing}')
    if not set(W).issubset(cost.keys()):
        missing = set(W) - set(cost.keys())
        raise ValueError(f'Missing cost rows for warehouses: {missing}')
    for w in W:
        if not set(C).issubset(cost[w].keys()):
            missing = set(C) - set(cost[w].keys())
            raise ValueError(f'Missing cost columns for warehouse {w}: {missing}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(w, c) for w in W for c in C], lb=0, vtype=GRB.CONTINUOUS, name='')
    obj_expr = gurobipy.quicksum((cost[w][c] * x[w, c] for w in W for c in C))
    m.setObjective(obj_expr, GRB.MINIMIZE)
    for c in C:
        expr = gurobipy.quicksum((x[w, c] for w in W))
        m.addConstr(expr == demand[c], name=f'DemandC_{c}')
    for w in W:
        expr = gurobipy.quicksum((x[w, c] for c in C))
        m.addConstr(expr <= capacity[w], name=f'SupplyW_{w}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(m.ObjVal)
        for var in m.getVars():
            print(var.VarName, var.X)
    else:
        print(m.Status)
    return m
m = solve_problem()