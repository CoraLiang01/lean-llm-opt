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
                    'matrix_table_id': 'file_2_view_0',
                    'row_axis': {'id_column': 'Unnamed: 0', 'table_id': 'file_1_view_0'},
                    'row_id_column': 'Unnamed: 0',
                    'type': 'matrix'}],
 'route': 'NRM',
 'tables': [{'aliases': ['file_0_view_0', 'demand.csv'],
             'columns': ['customer', 'demand'],
             'file_index': 0,
             'file_name': 'demand.csv',
             'filters': {},
             'original_rows': 5,
             'records': [{'source_row': 0, 'values': {'customer': 'C1', 'demand': '143'}},
                         {'source_row': 1, 'values': {'customer': 'C2', 'demand': '6'}},
                         {'source_row': 2, 'values': {'customer': 'C3', 'demand': '10'}},
                         {'source_row': 3, 'values': {'customer': 'C4', 'demand': '25'}},
                         {'source_row': 4, 'values': {'customer': 'C5', 'demand': '3'}}],
             'returned_rows': 5,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'},
            {'aliases': ['file_1_view_0', 'fixed_cost.csv'],
             'columns': ['Unnamed: 0', 'fixed_costs'],
             'file_index': 1,
             'file_name': 'fixed_cost.csv',
             'filters': {},
             'original_rows': 5,
             'records': [{'source_row': 0, 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '97.65000000000001'}},
                         {'source_row': 1, 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '99.76000000000001'}},
                         {'source_row': 2, 'values': {'Unnamed: 0': 'S3', 'fixed_costs': '100.76'}},
                         {'source_row': 3, 'values': {'Unnamed: 0': 'S4', 'fixed_costs': '105.32'}},
                         {'source_row': 4, 'values': {'Unnamed: 0': 'S5', 'fixed_costs': '98.88'}}],
             'returned_rows': 5,
             'role': 'descriptive non-unique role',
             'table_id': 'file_1_view_0'},
            {'aliases': ['file_2_view_0', 'transportation_costs.csv'],
             'columns': ['Unnamed: 0', 'C1', 'C2', 'C3', 'C4', 'C5'],
             'file_index': 2,
             'file_name': 'transportation_costs.csv',
             'filters': {},
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
             'role': 'descriptive non-unique role',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [{'column_ids_aligned': True,
                                   'expected_shape': [5, 5],
                                   'matrix_table_id': 'file_2_view_0',
                                   'row_ids_aligned': True,
                                   'shape': [5, 5]}],
                'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB
tables = {t['table_id']: t for t in CSVQA_DATA['tables']}
for t in CSVQA_DATA['tables']:
    for alias in t.get('aliases', []):
        tables[alias] = t
supplier_table = tables['file_1_view_0']
suppliers = []
fixed_costs = {}
for rec in supplier_table['records']:
    sid = rec['values']['Unnamed: 0']
    suppliers.append(sid)
    fixed_costs[sid] = float(rec['values']['fixed_costs'])
branch_table = tables['file_0_view_0']
branches = []
demand = {}
for rec in branch_table['records']:
    bid = rec['values']['customer']
    branches.append(bid)
    demand[bid] = float(rec['values']['demand'])
transport_table = tables['file_2_view_0']
transport_costs = {}
for rec in transport_table['records']:
    sid = rec['values']['Unnamed: 0']
    for bid in branches:
        transport_costs[sid, bid] = float(rec['values'][bid])
m = gp.Model('facility_location')
m.Params.MIPGap = 1e-06
y = {i: m.addVar(vtype=GRB.BINARY) for i in suppliers}
x = {(i, j): m.addVar(vtype=GRB.CONTINUOUS) for i in suppliers for j in branches}
for j in branches:
    m.addConstr(gp.quicksum((x[i, j] for i in suppliers)) == demand[j])
for i in suppliers:
    for j in branches:
        m.addConstr(x[i, j] <= demand[j] * y[i])
obj = gp.quicksum((fixed_costs[i] * y[i] for i in suppliers)) + gp.quicksum((transport_costs[i, j] * x[i, j] for i in suppliers for j in branches))
m.setObjective(obj, GRB.MINIMIZE)
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName} = {var.X}')
else:
    print(f'Status: {m.Status}')