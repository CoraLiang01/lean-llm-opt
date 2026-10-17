CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'On the Bandcamp sales platform, independent musicians and bands require inventory replenishment through '
          'warehouses. Multiple distribution warehouses, located in different cities, can provide the necessary '
          'inventory. Each warehouse incurs a fixed cost when starting operations, and the fixed cost data is provided '
          'in the “fixed_cost.csv” file. Each musician or band needs to source a certain quantity of goods from these '
          'warehouses. For each musician or band, the transportation cost per unit of goods from each warehouse is '
          "recorded in the “transportation_costs.csv” file. Demand information can be gained in 'demand.csv'. The "
          'objective is to determine which warehouses should be activated so that the demand of all musicians and '
          'bands is met while minimizing the total cost. The decision variables y_i are binary, indicating whether a '
          'warehouse is operational. The decision variables x_{ij} represent the quantity of goods that musician or '
          'band S_j sources from warehouse F_i. For each musician or band, x_{ij} represents the proportion of the '
          'total supply obtained from different warehouses.',
 'relationships': [{'column_axis': {'id_column': 'customer', 'table_id': 'file_0_view_0'},
                    'matrix_table_id': 'file_2_view_0',
                    'row_axis': {'id_column': 'Unnamed: 0', 'table_id': 'file_1_view_0'},
                    'row_id_column': 'Unnamed: 0',
                    'type': 'matrix'}],
 'route': 'NRM',
 'tables': [{'columns': ['customer', 'demand'],
             'file_index': 0,
             'file_name': 'demand.csv',
             'filters': {},
             'original_rows': 3,
             'records': [{'source_row': 0, 'values': {'customer': 'C1', 'demand': '1083'}},
                         {'source_row': 1, 'values': {'customer': 'C2', 'demand': '776'}},
                         {'source_row': 2, 'values': {'customer': 'C3', 'demand': '16214'}}],
             'returned_rows': 3,
             'role': 'descriptive non-unique role',
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
             'role': 'descriptive non-unique role',
             'table_id': 'file_1_view_0'},
            {'columns': ['Unnamed: 0', 'C1', 'C2', 'C3'],
             'file_index': 2,
             'file_name': 'transportation_costs.csv',
             'filters': {},
             'original_rows': 3,
             'records': [{'source_row': 0,
                          'values': {'C1': '1506.22', 'C2': '70.90000000000001', 'C3': '8.44', 'Unnamed: 0': 'S1'}},
                         {'source_row': 1,
                          'values': {'C1': '1732.65', 'C2': '1780.72', 'C3': '567.4400000000001', 'Unnamed: 0': 'S2'}},
                         {'source_row': 2,
                          'values': {'C1': '115.66', 'C2': '100.76', 'C3': '64.68000000000001', 'Unnamed: 0': 'S3'}}],
             'returned_rows': 3,
             'role': 'descriptive non-unique role',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [{'column_ids_aligned': True,
                                   'column_mapping_basis': 'exact',
                                   'expected_shape': [3, 3],
                                   'matrix_table_id': 'file_2_view_0',
                                   'row_ids_aligned': True,
                                   'row_mapping_basis': 'exact',
                                   'shape': [3, 3]}],
                'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']

    def get_table(table_id):
        for t in tables:
            if t['table_id'] == table_id:
                return t
        raise ValueError(f'Table {table_id} not found in CSVQA_DATA')
    table_fixed = get_table('file_1_view_0')
    table_demand = get_table('file_0_view_0')
    table_transport = get_table('file_2_view_0')
    I = [rec['values']['Unnamed: 0'] for rec in table_fixed['records']]
    J = [rec['values']['customer'] for rec in table_demand['records']]
    fixed_cost = {}
    for rec in table_fixed['records']:
        key = rec['values']['Unnamed: 0']
        val = rec['values']['fixed_costs']
        try:
            fixed_cost[key] = float(val)
        except Exception as e:
            raise ValueError(f'Invalid fixed_cost value for warehouse {key}: {val}') from e
    demand = {}
    for rec in table_demand['records']:
        key = rec['values']['customer']
        val = rec['values']['demand']
        try:
            demand[key] = float(val)
        except Exception as e:
            raise ValueError(f'Invalid demand value for customer {key}: {val}') from e
    transport_cost = {}
    for rec in table_transport['records']:
        i = rec['values']['Unnamed: 0']
        transport_cost[i] = {}
        for j in J:
            if j not in rec['values']:
                raise ValueError(f'Missing transport cost for warehouse {i} to customer {j}')
            val = rec['values'][j]
            try:
                transport_cost[i][j] = float(val)
            except Exception as e:
                raise ValueError(f'Invalid transport cost value for warehouse {i} to customer {j}: {val}') from e
    for i in I:
        if i not in fixed_cost:
            raise ValueError(f'Missing fixed_cost for warehouse {i}')
        if i not in transport_cost:
            raise ValueError(f'Missing transport_cost row for warehouse {i}')
        for j in J:
            if j not in transport_cost[i]:
                raise ValueError(f'Missing transport_cost for warehouse {i} to customer {j}')
    for j in J:
        if j not in demand:
            raise ValueError(f'Missing demand for customer {j}')
    m = gurobipy.Model('Warehouse_Opening')
    m.setParam('MIPGap', 1e-06)
    y = m.addVars(I, vtype=GRB.BINARY, name='')
    x = m.addVars([(i, j) for i in I for j in J], vtype=GRB.CONTINUOUS, lb=0, name='')
    obj = gurobipy.quicksum((fixed_cost[i] * y[i] for i in I)) + gurobipy.quicksum((transport_cost[i][j] * x[i, j] for i in I for j in J))
    m.setObjective(obj, GRB.MINIMIZE)
    m.addConstrs((gurobipy.quicksum((x[i, j] for i in I)) >= demand[j] for j in J), name='')
    m.addConstrs((x[i, j] <= demand[j] * y[i] for i in I for j in J), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()