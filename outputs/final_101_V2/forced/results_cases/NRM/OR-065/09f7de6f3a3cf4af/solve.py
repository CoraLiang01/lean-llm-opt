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
             'role': 'demand per customer',
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
             'role': 'fixed cost per warehouse',
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
             'role': 'transportation cost matrix',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [{'column_ids_aligned': True,
                                   'expected_shape': [3, 3],
                                   'matrix_table_id': 'file_2_view_0',
                                   'row_ids_aligned': True,
                                   'shape': [3, 3]}],
                'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB
import re
data = CSVQA_DATA
fixed_cost_table = next((t for t in data['tables'] if t['table_id'] == 'file_1_view_0'))
F = []
f = {}
for rec in fixed_cost_table['records']:
    i = rec['values']['Unnamed: 0']
    F.append(i)
    try:
        f[i] = float(rec['values']['fixed_costs'])
    except Exception:
        raise ValueError(f'Invalid fixed_costs for warehouse {i}')
demand_table = next((t for t in data['tables'] if t['table_id'] == 'file_0_view_0'))
S = []
d = {}
for rec in demand_table['records']:
    j = rec['values']['customer']
    S.append(j)
    try:
        d[j] = float(rec['values']['demand'])
    except Exception:
        raise ValueError(f'Invalid demand for customer {j}')
trans_cost_table = next((t for t in data['tables'] if t['table_id'] == 'file_2_view_0'))
c = {}
row_ids = [rec['values']['Unnamed: 0'] for rec in trans_cost_table['records']]
col_ids = [col for col in trans_cost_table['columns'] if col != 'Unnamed: 0']
if set(F) != set(row_ids):
    raise ValueError('Mismatch between warehouse ids in fixed_cost.csv and transportation_costs.csv')
if set(S) != set(col_ids):
    raise ValueError('Mismatch between customer ids in demand.csv and transportation_costs.csv')
for rec in trans_cost_table['records']:
    i = rec['values']['Unnamed: 0']
    for j in S:
        try:
            c[i, j] = float(rec['values'][j])
        except Exception:
            raise ValueError(f'Invalid transportation cost for warehouse {i}, customer {j}')
if set(F) != set(f.keys()):
    raise ValueError('Fixed cost data missing for some warehouses')
if set(S) != set(d.keys()):
    raise ValueError('Demand data missing for some customers')
for i in F:
    for j in S:
        if (i, j) not in c:
            raise ValueError(f'Transportation cost missing for warehouse {i}, customer {j}')
m = gp.Model('Bandcamp_Warehouse_Selection')
y = m.addVars(F, vtype=GRB.BINARY, name='')
x = m.addVars(F, S, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((f[i] * y[i] for i in F)) + gp.quicksum((c[i, j] * x[i, j] for i in F for j in S)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x[i, j] for i in F)) == d[j] for j in S), name='')
m.addConstrs((x[i, j] <= d[j] * y[i] for i in F for j in S), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')