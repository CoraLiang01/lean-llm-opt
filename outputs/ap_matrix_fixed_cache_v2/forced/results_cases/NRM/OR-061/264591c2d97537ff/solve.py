CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'In the Superstore chain, multiple branches require inventory replenishment, and several suppliers located '
          'in different cities can provide the necessary goods. Each supplier incurs a fixed cost upon starting '
          'operations, with the fixed cost data provided in the “fixed_cost.csv” file. Each branch needs to source a '
          'certain quantity of goods from these suppliers. For each branch, the transportation cost per unit of goods '
          'from each supplier is recorded in the “transportation_costs.csv” file. Demand information can be gained in '
          "'demand.csv'. The objective is to determine which suppliers to activate so that the demand of all branches "
          'is met while minimizing the total cost. The decision variables y_i are binary, indicating whether a '
          'supplier is operational (open). The decision variables x_{ij} represent the quantity of goods that branch '
          'S_j sources from supplier F_i. For each branch, x_{ij} represents the proportion of the total supply '
          'obtained from different suppliers. These decision variables help determine the optimal allocation of supply '
          'to minimize the total of fixed and transportation costs.',
 'relationships': [{'column_axis': {'id_column': 'customer', 'table_id': 'file_0_view_0'},
                    'matrix_column_id_map': {},
                    'matrix_format': 'wide',
                    'matrix_metadata_columns': [],
                    'matrix_table_id': 'file_2_view_0',
                    'matrix_value_columns': ['C1', 'C2', 'C3', 'C4', 'C5'],
                    'row_axis': {'id_column': 'Unnamed: 0', 'table_id': 'file_1_view_0'},
                    'row_id_column': 'Unnamed: 0',
                    'type': 'matrix'}],
 'route': 'NRM',
 'tables': [{'columns': ['customer', 'demand'],
             'file_index': 0,
             'file_name': 'demand.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0, 'values': {'customer': 'C1', 'demand': '143'}},
                         {'source_row': 1, 'values': {'customer': 'C2', 'demand': '6'}},
                         {'source_row': 2, 'values': {'customer': 'C3', 'demand': '10'}},
                         {'source_row': 3, 'values': {'customer': 'C4', 'demand': '25'}},
                         {'source_row': 4, 'values': {'customer': 'C5', 'demand': '3'}}],
             'returned_rows': 5,
             'role': 'branch demand',
             'table_id': 'file_0_view_0'},
            {'columns': ['Unnamed: 0', 'fixed_costs'],
             'file_index': 1,
             'file_name': 'fixed_cost.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0, 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '97.65000000000001'}},
                         {'source_row': 1, 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '99.76000000000001'}},
                         {'source_row': 2, 'values': {'Unnamed: 0': 'S3', 'fixed_costs': '100.76'}},
                         {'source_row': 3, 'values': {'Unnamed: 0': 'S4', 'fixed_costs': '105.32'}},
                         {'source_row': 4, 'values': {'Unnamed: 0': 'S5', 'fixed_costs': '98.88'}}],
             'returned_rows': 5,
             'role': 'supplier fixed cost',
             'table_id': 'file_1_view_0'},
            {'columns': ['Unnamed: 0', 'C1', 'C2', 'C3', 'C4', 'C5'],
             'file_index': 2,
             'file_name': 'transportation_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0,
                          'values': {'C1': '150.74',
                                     'C2': '0.02',
                                     'C3': '49.13',
                                     'C4': '2080.15',
                                     'C5': '426.4',
                                     'Unnamed: 0': 'S1'}},
                         {'source_row': 1,
                          'values': {'C1': '233.05',
                                     'C2': '97.73',
                                     'C3': '49.84',
                                     'C4': '1982.39',
                                     'C5': '23.96',
                                     'Unnamed: 0': 'S2'}},
                         {'source_row': 2,
                          'values': {'C1': '55.68',
                                     'C2': '935.61',
                                     'C3': '4.03',
                                     'C4': '73.09',
                                     'C5': '525.3200000000001',
                                     'Unnamed: 0': 'S3'}},
                         {'source_row': 3,
                          'values': {'C1': '1483.82',
                                     'C2': '1801.08',
                                     'C3': '112.16',
                                     'C4': '816.05',
                                     'C5': '107.01',
                                     'Unnamed: 0': 'S4'}},
                         {'source_row': 4,
                          'values': {'C1': '1119.47',
                                     'C2': '884.3099999999999',
                                     'C3': '0.08',
                                     'C4': '1544.95',
                                     'C5': '543.67',
                                     'Unnamed: 0': 'S5'}}],
             'returned_rows': 5,
             'role': 'transportation cost matrix',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [{'column_ids_aligned': True,
                                   'column_order_matches': True,
                                   'column_permutation': [0, 1, 2, 3, 4],
                                   'expected_shape': [5, 5],
                                   'format': 'wide',
                                   'matrix_table_id': 'file_2_view_0',
                                   'metadata_columns': [],
                                   'row_ids_aligned': True,
                                   'row_order_matches': True,
                                   'row_permutation': [0, 1, 2, 3, 4],
                                   'shape': [5, 5],
                                   'value_columns': ['C1', 'C2', 'C3', 'C4', 'C5']}],
                'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB
data = CSVQA_DATA
suppliers = []
fixed_costs = {}
for rec in [r['values'] for r in data['tables'] if r['table_id'] == 'file_1_view_0'][0:][0]['records'] if False else [r['values'] for r in data['tables'] if r['table_id'] == 'file_1_view_0'][0:][0]['records']:
    pass
for rec in [r['values'] for r in data['tables'] if r['table_id'] == 'file_1_view_0'][0:][0]['records'] if False else [r['values'] for r in data['tables'] if r['table_id'] == 'file_1_view_0'][0]['records']:
    sid = rec['Unnamed: 0']
    suppliers.append(sid)
    try:
        fixed_costs[sid] = float(rec['fixed_costs'])
    except Exception:
        raise ValueError(f'Invalid fixed_costs for supplier {sid}')
branches = []
demands = {}
for rec in [r['values'] for r in data['tables'] if r['table_id'] == 'file_0_view_0'][0]['records']:
    bid = rec['customer']
    branches.append(bid)
    try:
        demands[bid] = float(rec['demand'])
    except Exception:
        raise ValueError(f'Invalid demand for branch {bid}')
costs = {}
cost_table = None
for t in data['tables']:
    if t['table_id'] == 'file_2_view_0':
        cost_table = t
        break
if cost_table is None:
    raise ValueError('Transportation cost table not found')
for rec in cost_table['records']:
    row = rec['values']
    sid = row['Unnamed: 0']
    for bid in branches:
        if bid not in row:
            raise ValueError(f'Missing cost for supplier {sid}, branch {bid}')
        try:
            costs[sid, bid] = float(row[bid])
        except Exception:
            raise ValueError(f'Invalid cost for supplier {sid}, branch {bid}')
if set(suppliers) != set([rec['values']['Unnamed: 0'] for rec in cost_table['records']]):
    raise ValueError('Supplier indices mismatch between fixed_costs and transportation_costs')
if set(branches) != set(cost_table['columns'][1:]):
    raise ValueError('Branch indices mismatch between demand and transportation_costs')
M = sum((demands[bid] for bid in branches))
M_i = {sid: M for sid in suppliers}
m = gp.Model('Superstore_Supplier_Activation')
y_vars = m.addVars(suppliers, vtype=GRB.BINARY, name='')
x_vars = m.addVars(suppliers, branches, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((fixed_costs[sid] * y_vars[sid] for sid in suppliers)) + gp.quicksum((costs[sid, bid] * x_vars[sid, bid] for sid in suppliers for bid in branches)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[sid, bid] for sid in suppliers)) == demands[bid] for bid in branches), name='')
m.addConstrs((gp.quicksum((x_vars[sid, bid] for bid in branches)) <= M_i[sid] * y_vars[sid] for sid in suppliers), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')