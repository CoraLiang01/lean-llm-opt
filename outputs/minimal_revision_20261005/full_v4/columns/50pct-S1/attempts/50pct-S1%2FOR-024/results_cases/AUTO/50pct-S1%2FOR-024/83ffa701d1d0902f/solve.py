CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'On the Bandcamp sales platform, independent musicians and bands require inventory replenishment through '
          'warehouses. Multiple distribution warehouses, located in different cities, can provide the necessary '
          'inventory. Each warehouse incurs a fixed cost when starting operations, and the fixed cost data is provided '
          'in the ‚Äúfixed_cost.csv‚Äù file. Each musician or band needs to source a certain quantity of goods from '
          'these warehouses. For each musician or band, the transportation cost per unit of goods from each warehouse '
          "is recorded in the ‚Äútransportation_costs.csv‚Äù file. Demand information can be gained in 'demand.csv'. "
          'The objective is to determine which warehouses should be activated so that the demand of all musicians and '
          'bands is met while minimizing the total cost. The decision variables y_i are binary, indicating whether a '
          'warehouse is operational. The decision variables x_{ij} represent the quantity of goods that musician or '
          'band S_j sources from warehouse F_i. For each musician or band, x_{ij} represents the proportion of the '
          'total supply obtained from different warehouses.',
 'relationships': [{'column_axis': {'id_column': 'customer', 'table_id': 'file_0_view_0'},
                    'matrix_table_id': 'file_2_view_0',
                    'row_axis': {'id_column': 'Unnamed: 0', 'table_id': 'file_1_view_0'},
                    'row_id_column': 'Unnamed: 0',
                    'type': 'matrix'}],
 'route': 'FLP',
 'tables': [{'columns': ['customer', 'demand'],
             'file_index': 0,
             'file_name': 'demand.csv',
             'filters': {},
             'original_rows': 3,
             'records': [{'source_row': 0, 'values': {'customer': 'C1', 'demand': '1083'}},
                         {'source_row': 1, 'values': {'customer': 'C2', 'demand': '776'}},
                         {'source_row': 2, 'values': {'customer': 'C3', 'demand': '16214'}}],
             'returned_rows': 3,
             'role': 'musician/band demand',
             'table_id': 'file_0_view_0'},
            {'columns': ['Unnamed: 0', 'fixed_costs'],
             'file_index': 1,
             'file_name': 'fixed_cost.csv',
             'filters': {},
             'original_rows': 3,
             'records': [{'source_row': 0, 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '102.33'}},
                         {'source_row': 1, 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '94.92'}},
                         {'source_row': 2, 'values': {'Unnamed: 0': 'S3', 'fixed_costs': '91.83'}}],
             'returned_rows': 3,
             'role': 'warehouse fixed cost',
             'table_id': 'file_1_view_0'},
            {'columns': ['Unnamed: 0', 'C1', 'C2', 'C3'],
             'file_index': 2,
             'file_name': 'transportation_costs.csv',
             'filters': {},
             'original_rows': 3,
             'records': [{'source_row': 0, 'values': {'C1': '1506.22', 'C2': '70.9', 'C3': '8.44', 'Unnamed: 0': 'S1'}},
                         {'source_row': 1,
                          'values': {'C1': '1732.65', 'C2': '1780.72', 'C3': '567.44', 'Unnamed: 0': 'S2'}},
                         {'source_row': 2,
                          'values': {'C1': '115.66', 'C2': '100.76', 'C3': '64.68', 'Unnamed: 0': 'S3'}}],
             'returned_rows': 3,
             'role': 'transportation cost matrix',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [{'column_ids_aligned': True,
                                   'column_mapping_basis': 'exact',
                                   'expected_shape': [3, 3],
                                   'matrix_table_id': 'file_2_view_0',
                                   'row_ids_aligned': True,
                                   'row_mapping_basis': 'exact',
                                   'shape': [3, 3]}],
                'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    warehouses = [rec['values']['Unnamed: 0'] for rec in CSVQA_DATA['tables'][1]['records']]
    customers = [rec['values']['customer'] for rec in CSVQA_DATA['tables'][0]['records']]
    demand = {}
    for rec in CSVQA_DATA['tables'][0]['records']:
        j = rec['values']['customer']
        d = rec['values']['demand']
        demand[j] = float(d)
    fixed_cost = {}
    for rec in CSVQA_DATA['tables'][1]['records']:
        i = rec['values']['Unnamed: 0']
        f = rec['values']['fixed_costs']
        fixed_cost[i] = float(f)
    cost = {}
    for rec in CSVQA_DATA['tables'][2]['records']:
        i = rec['values']['Unnamed: 0']
        cost[i] = {}
        for j in customers:
            cij = rec['values'][j]
            cost[i][j] = float(cij)
    M = sum((demand[j] for j in customers))
    for i in warehouses:
        if i not in fixed_cost:
            raise ValueError(f'Missing fixed cost for warehouse {i}')
        if i not in cost:
            raise ValueError(f'Missing cost row for warehouse {i}')
        for j in customers:
            if j not in cost[i]:
                raise ValueError(f'Missing transportation cost for warehouse {i}, customer {j}')
    for j in customers:
        if j not in demand:
            raise ValueError(f'Missing demand for customer {j}')
    m = gp.Model('Bandcamp_FLP')
    m.Params.MIPGap = 0.0001
    x = m.addVars(warehouses, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
    y = m.addVars(warehouses, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in warehouses for j in customers)) + gp.quicksum((fixed_cost[i] * y[i] for i in warehouses)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in warehouses)) == demand[j] for j in customers), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in customers)) <= M * y[i] for i in warehouses), name='')
    m.optimize()
    return m
m = solve_problem()