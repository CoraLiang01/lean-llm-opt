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
                    'matrix_column_id_map': {},
                    'matrix_format': 'wide',
                    'matrix_metadata_columns': [],
                    'matrix_table_id': 'file_2_view_0',
                    'matrix_value_columns': ['C1', 'C2', 'C3'],
                    'row_axis': {'id_column': 'Unnamed: 0', 'table_id': 'file_1_view_0'},
                    'row_id_column': 'Unnamed: 0',
                    'type': 'matrix'}],
 'route': 'NRM',
 'tables': [{'columns': ['customer', 'demand'],
             'file_index': 0,
             'file_name': 'demand.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 3,
             'records': [{'source_row': 0, 'values': {'customer': 'C1', 'demand': '1083'}},
                         {'source_row': 1, 'values': {'customer': 'C2', 'demand': '776'}},
                         {'source_row': 2, 'values': {'customer': 'C3', 'demand': '16214'}}],
             'returned_rows': 3,
             'role': 'demand per customer',
             'table_id': 'file_0_view_0'},
            {'columns': ['Unnamed: 0', 'fixed_costs'],
             'file_index': 1,
             'file_name': 'fixed_cost.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 3,
             'records': [{'source_row': 0, 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '102.33'}},
                         {'source_row': 1, 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '94.92'}},
                         {'source_row': 2, 'values': {'Unnamed: 0': 'S3', 'fixed_costs': '91.83'}}],
             'returned_rows': 3,
             'role': 'fixed cost per warehouse',
             'table_id': 'file_1_view_0'},
            {'columns': ['Unnamed: 0', 'C1', 'C2', 'C3'],
             'file_index': 2,
             'file_name': 'transportation_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 3,
             'records': [{'source_row': 0,
                          'values': {'C1': '1506.22', 'C2': '70.90000000000001', 'C3': '8.44', 'Unnamed: 0': 'S1'}},
                         {'source_row': 1,
                          'values': {'C1': '1732.65', 'C2': '1780.72', 'C3': '567.4400000000001', 'Unnamed: 0': 'S2'}},
                         {'source_row': 2,
                          'values': {'C1': '115.66', 'C2': '100.76', 'C3': '64.68000000000001', 'Unnamed: 0': 'S3'}}],
             'returned_rows': 3,
             'role': 'transportation cost matrix',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [{'column_ids_aligned': True,
                                   'column_order_matches': True,
                                   'column_permutation': [0, 1, 2],
                                   'expected_shape': [3, 3],
                                   'format': 'wide',
                                   'matrix_table_id': 'file_2_view_0',
                                   'metadata_columns': [],
                                   'row_ids_aligned': True,
                                   'row_order_matches': True,
                                   'row_permutation': [0, 1, 2],
                                   'shape': [3, 3],
                                   'value_columns': ['C1', 'C2', 'C3']}],
                'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB
tables = CSVQA_DATA['tables']
warehouse_table = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
warehouses = [rec['values']['Unnamed: 0'] for rec in warehouse_table['records']]
fixed_costs = {}
for rec in warehouse_table['records']:
    i = rec['values']['Unnamed: 0']
    try:
        fixed_costs[i] = float(rec['values']['fixed_costs'])
    except Exception:
        raise ValueError(f'Invalid fixed_costs for warehouse {i}')
customer_table = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
customers = [rec['values']['customer'] for rec in customer_table['records']]
demands = {}
for rec in customer_table['records']:
    j = rec['values']['customer']
    try:
        demands[j] = float(rec['values']['demand'])
    except Exception:
        raise ValueError(f'Invalid demand for customer {j}')
transport_table = next((t for t in tables if t['table_id'] == 'file_2_view_0'))
transport_costs = {}
for rec in transport_table['records']:
    i = rec['values']['Unnamed: 0']
    transport_costs[i] = {}
    for j in customers:
        try:
            transport_costs[i][j] = float(rec['values'][j])
        except Exception:
            raise ValueError(f'Missing or invalid transportation cost for warehouse {i}, customer {j}')
if set(fixed_costs.keys()) != set(warehouses):
    raise ValueError('Mismatch in warehouse identifiers between fixed_costs and warehouse list')
if set(demands.keys()) != set(customers):
    raise ValueError('Mismatch in customer identifiers between demands and customer list')
for i in warehouses:
    if i not in transport_costs:
        raise ValueError(f'Missing transportation cost row for warehouse {i}')
    if set(transport_costs[i].keys()) != set(customers):
        raise ValueError(f'Mismatch in transportation cost columns for warehouse {i}')
M = sum((demands[j] for j in customers))
M_i = {i: M for i in warehouses}
m = gp.Model('Bandcamp_Warehouse_Selection')
y_vars = m.addVars(warehouses, vtype=GRB.BINARY, name='')
x_vars = m.addVars(warehouses, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((fixed_costs[i] * y_vars[i] for i in warehouses)) + gp.quicksum((transport_costs[i][j] * x_vars[i, j] for i in warehouses for j in customers)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[i, j] for i in warehouses)) == demands[j] for j in customers), name='')
m.addConstrs((gp.quicksum((x_vars[i, j] for j in customers)) <= M_i[i] * y_vars[i] for i in warehouses), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')