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
 'route': 'NRM',
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
                                   'column_order_matches': True,
                                   'expected_shape': [5, 5],
                                   'matrix_table_id': 'file_2_view_0',
                                   'row_ids_aligned': True,
                                   'row_order_matches': True,
                                   'shape': [5, 5]}],
                'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB
data = CSVQA_DATA
supply_table = next((t for t in data['tables'] if t['table_id'] == 'file_1_view_0'))
S = [rec['values']['region'] for rec in supply_table['records']]
a_s = {rec['values']['region']: float(rec['values']['supply_capacity']) for rec in supply_table['records']}
demand_table = next((t for t in data['tables'] if t['table_id'] == 'file_0_view_0'))
D = [rec['values']['customer'] for rec in demand_table['records']]
b_d = {rec['values']['customer']: float(rec['values']['demand']) for rec in demand_table['records']}
cost_table = next((t for t in data['tables'] if t['table_id'] == 'file_2_view_0'))
cost_rows = cost_table['records']
cost_columns = [col for col in cost_table['columns'] if col != 'Unnamed: 0']
if cost_columns != D:
    raise ValueError('Transportation cost columns do not match store identifiers D.')
if [row['values']['Unnamed: 0'] for row in cost_rows] != S:
    raise ValueError('Transportation cost rows do not match warehouse identifiers S.')
c_sd = {}
for row in cost_rows:
    s = row['values']['Unnamed: 0']
    for d in D:
        c_sd[s, d] = float(row['values'][d])
if set(a_s.keys()) != set(S):
    raise ValueError('Supply capacity keys do not match warehouse set S.')
if set(b_d.keys()) != set(D):
    raise ValueError('Demand keys do not match store set D.')
for s in S:
    for d in D:
        if (s, d) not in c_sd:
            raise ValueError(f'Missing transportation cost for ({s},{d})')
m = gp.Model('GreenMart_Transportation')
x_vars = m.addVars(S, D, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((c_sd[s, d] * x_vars[s, d] for s in S for d in D)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[s, d] for s in S)) == b_d[d] for d in D), name='')
m.addConstrs((gp.quicksum((x_vars[s, d] for d in D)) <= a_s[s] for s in S), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')