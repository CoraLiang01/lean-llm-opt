CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'In the context of Colorado Motor Vehicle Sales, multiple car dealerships require inventory, and several '
          'suppliers located in different cities can provide the necessary vehicles. Each supplier incurs a fixed cost '
          'upon starting operations, with the fixed cost data provided in “fixed_cost.csv.” Each dealership needs to '
          'source a certain number of vehicles from these suppliers. For each dealership, the transportation cost per '
          'vehicle from each supplier is recorded in “transportation_costs.csv.” Demand information can be gained in '
          "'demand.csv'. The objective is to determine which suppliers to open so that the demand of all dealerships "
          'is met while minimizing the total cost. The decision variables y_i are binary, indicating whether a '
          'supplier is operational (open). The decision variables x_{ij} represent the quantity of vehicles that '
          'dealership S_j sources from supplier F_i. For each dealership, x_{ij} represents the proportion of the '
          'total supply obtained from different suppliers.',
 'relationships': [{'column_axis': {'id_column': 'customer', 'table_id': 'file_0_view_0'},
                    'matrix_table_id': 'file_2_view_0',
                    'row_axis': {'id_column': 'Unnamed: 0', 'table_id': 'file_1_view_0'},
                    'row_id_column': 'Unnamed: 0',
                    'type': 'matrix'}],
 'route': 'RA',
 'tables': [{'aliases': ['file_0_view_0', 'demand.csv'],
             'columns': ['customer', 'demand'],
             'file_index': 0,
             'file_name': 'demand.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 9,
             'records': [{'source_row': 0, 'values': {'customer': 'C1', 'demand': '4742532000'}},
                         {'source_row': 1, 'values': {'customer': 'C2', 'demand': '1600594000'}},
                         {'source_row': 2, 'values': {'customer': 'C3', 'demand': '5086889000'}},
                         {'source_row': 3, 'values': {'customer': 'C4', 'demand': '1027326000'}},
                         {'source_row': 4, 'values': {'customer': 'C5', 'demand': '11926044000'}},
                         {'source_row': 5, 'values': {'customer': 'C6', 'demand': '9058407000'}},
                         {'source_row': 6, 'values': {'customer': 'C7', 'demand': '5344367000'}},
                         {'source_row': 7, 'values': {'customer': 'C8', 'demand': '677201000'}},
                         {'source_row': 8, 'values': {'customer': 'C9', 'demand': '3236493000'}}],
             'returned_rows': 9,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'},
            {'aliases': ['file_1_view_0', 'fixed_cost.csv'],
             'columns': ['Unnamed: 0', 'fixed_costs'],
             'file_index': 1,
             'file_name': 'fixed_cost.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 8,
             'records': [{'source_row': 0, 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '100.64'}},
                         {'source_row': 1, 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '98.72'}},
                         {'source_row': 2, 'values': {'Unnamed: 0': 'S3', 'fixed_costs': '100.18'}},
                         {'source_row': 3, 'values': {'Unnamed: 0': 'S4', 'fixed_costs': '96.58'}},
                         {'source_row': 4, 'values': {'Unnamed: 0': 'S5', 'fixed_costs': '95.75'}},
                         {'source_row': 5, 'values': {'Unnamed: 0': 'S6', 'fixed_costs': '99.06'}},
                         {'source_row': 6, 'values': {'Unnamed: 0': 'S7', 'fixed_costs': '101.78'}},
                         {'source_row': 7, 'values': {'Unnamed: 0': 'S8', 'fixed_costs': '93.86'}}],
             'returned_rows': 8,
             'role': 'descriptive non-unique role',
             'table_id': 'file_1_view_0'},
            {'aliases': ['file_2_view_0', 'transportation_costs.csv'],
             'columns': ['Unnamed: 0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'],
             'file_index': 2,
             'file_name': 'transportation_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 8,
             'records': [{'source_row': 0,
                          'values': {'C1': '1091.04',
                                     'C2': '85.72',
                                     'C3': '99.08',
                                     'C4': '747.35',
                                     'C5': '893.86',
                                     'C6': '23.65',
                                     'C7': '15.11',
                                     'C8': '15.03',
                                     'C9': '497.88',
                                     'Unnamed: 0': 'S1'}},
                         {'source_row': 1,
                          'values': {'C1': '58.88',
                                     'C2': '1617.16',
                                     'C3': '1786.44',
                                     'C4': '951.81',
                                     'C5': '56.45',
                                     'C6': '642.77',
                                     'C7': '16.69',
                                     'C8': '0.63',
                                     'C9': '11.2',
                                     'Unnamed: 0': 'S2'}},
                         {'source_row': 2,
                          'values': {'C1': '110.47',
                                     'C2': '0.04',
                                     'C3': '38.89',
                                     'C4': '1397.95',
                                     'C5': '2361.45',
                                     'C6': '107.62',
                                     'C7': '1598.5',
                                     'C8': '76.41',
                                     'C9': '1382.84',
                                     'Unnamed: 0': 'S3'}},
                         {'source_row': 3,
                          'values': {'C1': '1458.85',
                                     'C2': '1049.27',
                                     'C3': '597.32',
                                     'C4': '1731.9',
                                     'C5': '69.09',
                                     'C6': '1227.17',
                                     'C7': '1187.55',
                                     'C8': '1017.16',
                                     'C9': '52.15',
                                     'Unnamed: 0': 'S4'}},
                         {'source_row': 4,
                          'values': {'C1': '0.38',
                                     'C2': '2315.52',
                                     'C3': '1313.06',
                                     'C4': '1253.71',
                                     'C5': '50.24',
                                     'C6': '29.19',
                                     'C7': '60.17',
                                     'C8': '1077.35',
                                     'C9': '70.11',
                                     'Unnamed: 0': 'S5'}},
                         {'source_row': 5,
                          'values': {'C1': '58.2',
                                     'C2': '1395.81',
                                     'C3': '84.6',
                                     'C4': '830.64',
                                     'C5': '1003.86',
                                     'C6': '631.17',
                                     'C7': '31.13',
                                     'C8': '1.4',
                                     'C9': '246.24',
                                     'Unnamed: 0': 'S6'}},
                         {'source_row': 6,
                          'values': {'C1': '1255.23',
                                     'C2': '1382.31',
                                     'C3': '78.79',
                                     'C4': '829.02',
                                     'C5': '67.31',
                                     'C6': '877.35',
                                     'C7': '185.28',
                                     'C8': '221.98',
                                     'C9': '0.05',
                                     'Unnamed: 0': 'S7'}},
                         {'source_row': 7,
                          'values': {'C1': '1990.09',
                                     'C2': '1.23',
                                     'C3': '38.97',
                                     'C4': '1396.35',
                                     'C5': '112.54',
                                     'C6': '107.54',
                                     'C7': '1596.74',
                                     'C8': '76.32',
                                     'C9': '1183.79',
                                     'Unnamed: 0': 'S8'}}],
             'returned_rows': 8,
             'role': 'descriptive non-unique role',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [{'column_ids_aligned': True,
                                   'expected_shape': [8, 9],
                                   'matrix_table_id': 'file_2_view_0',
                                   'row_ids_aligned': True,
                                   'shape': [8, 9]}],
                'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB
tables = CSVQA_DATA['tables']

def get_table(table_id_or_alias):
    for t in tables:
        if t['table_id'] == table_id_or_alias or table_id_or_alias in t.get('aliases', []):
            return t
    raise KeyError(f'Table {table_id_or_alias} not found')
demand_table = get_table('file_0_view_0')
customers = []
demand = {}
for rec in demand_table['records']:
    cust = rec['values']['customer']
    dem = int(rec['values']['demand'])
    customers.append(cust)
    demand[cust] = dem
fixed_cost_table = get_table('file_1_view_0')
suppliers = []
fixed_costs = {}
for rec in fixed_cost_table['records']:
    sup = rec['values']['Unnamed: 0']
    fc = float(rec['values']['fixed_costs'])
    suppliers.append(sup)
    fixed_costs[sup] = fc
trans_table = get_table('file_2_view_0')
c_ij = {}
for rec in trans_table['records']:
    sup = rec['values']['Unnamed: 0']
    c_ij[sup] = {}
    for cust in customers:
        key = f'C{cust[1:]}'
        c_ij[sup][cust] = float(rec['values'][key])
m = gp.Model('supplier_selection')
y_vars = {}
for s in suppliers:
    y_vars[s] = m.addVar(vtype=GRB.BINARY)
x_vars = {}
for s in suppliers:
    for c in customers:
        x_vars[s, c] = m.addVar(vtype=GRB.INTEGER)
m.update()
for c in customers:
    m.addConstr(gp.quicksum((x_vars[s, c] for s in suppliers)) == demand[c])
for s in suppliers:
    for c in customers:
        m.addConstr(x_vars[s, c] <= demand[c] * y_vars[s])
obj = gp.quicksum((fixed_costs[s] * y_vars[s] for s in suppliers)) + gp.quicksum((c_ij[s][c] * x_vars[s, c] for s in suppliers for c in customers))
m.setObjective(obj, GRB.MINIMIZE)
m.Params.MIPGap = 1e-06
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} = {v.X}')
else:
    print(f'Solver status: {m.Status}')