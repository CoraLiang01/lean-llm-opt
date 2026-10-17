CSVQA_DATA = {'ignored_file_indices': [],
 'query': '“BrewCo,” a beverage manufacturer, operates multiple production facilities that distribute drinks to '
          'various retail locations. The daily demand for each retail outlet is specified in “customer_demand.csv,” '
          'while the production capacity of each plant is outlined in “supply_capacity.csv.” The transportation cost '
          'per unit of beverages from each plant to each outlet is recorded in “transportation_costs.csv.” The goal is '
          'to determine the optimal quantity of beverages to be shipped from each production plant to each retail '
          'outlet, ensuring all outlet demands are met without surpassing any plant’s production capacity, while '
          'minimizing the total transportation cost.',
 'relationships': [{'column_axis': {'id_column': 'customer', 'table_id': 'file_0_view_0'},
                    'matrix_table_id': 'file_2_view_0',
                    'row_axis': {'id_column': 'Unnamed: 0', 'table_id': 'file_1_view_0'},
                    'row_id_column': 'Unnamed: 0',
                    'type': 'matrix'}],
 'route': 'TP',
 'tables': [{'columns': ['customer', 'demand'],
             'file_index': 0,
             'file_name': 'customer_demand.csv',
             'filters': {},
             'original_rows': 4,
             'records': [{'source_row': 0, 'values': {'customer': 'C1', 'demand': '94'}},
                         {'source_row': 1, 'values': {'customer': 'C2', 'demand': '39'}},
                         {'source_row': 2, 'values': {'customer': 'C3', 'demand': '65'}},
                         {'source_row': 3, 'values': {'customer': 'C4', 'demand': '435'}}],
             'returned_rows': 4,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'},
            {'columns': ['Unnamed: 0', 'supply_capacity'],
             'file_index': 1,
             'file_name': 'supply_capacity.csv',
             'filters': {},
             'original_rows': 4,
             'records': [{'source_row': 0, 'values': {'Unnamed: 0': 'S1', 'supply_capacity': '2531'}},
                         {'source_row': 1, 'values': {'Unnamed: 0': 'S2', 'supply_capacity': '20'}},
                         {'source_row': 2, 'values': {'Unnamed: 0': 'S3', 'supply_capacity': '210'}},
                         {'source_row': 3, 'values': {'Unnamed: 0': 'S4', 'supply_capacity': '241'}}],
             'returned_rows': 4,
             'role': 'descriptive non-unique role',
             'table_id': 'file_1_view_0'},
            {'columns': ['Unnamed: 0', 'C1', 'C2', 'C3', 'C4'],
             'file_index': 2,
             'file_name': 'transportation_costs.csv',
             'filters': {},
             'original_rows': 4,
             'records': [{'source_row': 0,
                          'values': {'C1': '543.756480860856',
                                     'C2': '23.685276141764653',
                                     'C3': '23.676386730773032',
                                     'C4': '447.75143678673766',
                                     'Unnamed: 0': 'S1'}},
                         {'source_row': 1,
                          'values': {'C1': '883.9151090405642',
                                     'C2': '0.04977684765576961',
                                     'C3': '0.0350986687216299',
                                     'C4': '44.45588531711622',
                                     'Unnamed: 0': 'S2'}},
                         {'source_row': 2,
                          'values': {'C1': '537.3456896658107',
                                     'C2': '23.769274659075112',
                                     'C3': '498.95659249465467',
                                     'C4': '440.60737890439776',
                                     'Unnamed: 0': 'S3'}},
                         {'source_row': 3,
                          'values': {'C1': '1791.493192397229',
                                     'C2': '68.21633865655126',
                                     'C3': '1432.4837339656747',
                                     'C4': '1527.7635425462734',
                                     'Unnamed: 0': 'S4'}}],
             'returned_rows': 4,
             'role': 'descriptive non-unique role',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [{'column_ids_aligned': True,
                                   'column_mapping_basis': 'exact',
                                   'expected_shape': [4, 4],
                                   'matrix_table_id': 'file_2_view_0',
                                   'row_ids_aligned': True,
                                   'row_mapping_basis': 'exact',
                                   'shape': [4, 4]}],
                'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    demand_table = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    supply_table = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
    cost_table = next((t for t in tables if t['table_id'] == 'file_2_view_0'))
    I = set()
    for record in supply_table['records']:
        supplier = record['values']['Unnamed: 0']
        I.add(supplier)
    J = set()
    for record in demand_table['records']:
        customer = record['values']['customer']
        J.add(customer)
    demand = {}
    for record in demand_table['records']:
        customer = record['values']['customer']
        demand[customer] = float(record['values']['demand'])
    supply = {}
    for record in supply_table['records']:
        supplier = record['values']['Unnamed: 0']
        supply[supplier] = float(record['values']['supply_capacity'])
    cost = {}
    for record in cost_table['records']:
        supplier = record['values']['Unnamed: 0']
        cost[supplier] = {}
        for col in cost_table['columns']:
            if col == 'Unnamed: 0':
                continue
            cost[supplier][col] = float(record['values'][col])
    for supplier in I:
        if supplier not in supply:
            raise ValueError(f'Missing supply capacity for supplier {supplier}')
        if supplier not in cost:
            raise ValueError(f'Missing cost row for supplier {supplier}')
    for customer in J:
        if customer not in demand:
            raise ValueError(f'Missing demand for customer {customer}')
    for supplier in I:
        for customer in J:
            if customer not in cost[supplier]:
                raise ValueError(f'Missing cost for supplier {supplier} to customer {customer}')
    m = gurobipy.Model('TransportationProblem')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(i, j) for i in I for j in J], lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gurobipy.quicksum((cost[i][j] * x[i, j] for i in I for j in J)), GRB.MINIMIZE)
    for j in J:
        m.addConstr(gurobipy.quicksum((x[i, j] for i in I)) >= demand[j], name=f'demand_{j}')
    for i in I:
        m.addConstr(gurobipy.quicksum((x[i, j] for j in J)) <= supply[i], name=f'supply_{i}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for variable in m.getVars():
            print(f'{variable.VarName}: {variable.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()