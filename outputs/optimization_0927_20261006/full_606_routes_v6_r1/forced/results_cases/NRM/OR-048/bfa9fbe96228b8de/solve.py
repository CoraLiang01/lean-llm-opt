CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'Amazon needs to allocate different types of air conditioners into different warehouse storage areas. '
          'Specifically, Amazon has several storage areas, each with a capacity limit provided in “capacity.csv.” The '
          'predefined value and size of each air conditioner type can be found in “products.csv.” The objective is to '
          'determine the optimal number of units of each air conditioner type to place in each storage area to '
          'maximize the total value of the air conditioners across all areas, while ensuring that the total size of '
          'the units in each area does not exceed its capacity. The decision variablesx_ijrepresent the number of '
          'units of air conditioner type j to be placed in storage area i.The decision variables must be integers.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['StorageID', 'Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {},
             'original_rows': 15,
             'records': [{'source_row': 0, 'values': {'Capacity': '1083', 'StorageID': '1'}},
                         {'source_row': 1, 'values': {'Capacity': '1840', 'StorageID': '2'}},
                         {'source_row': 2, 'values': {'Capacity': '770', 'StorageID': '3'}},
                         {'source_row': 3, 'values': {'Capacity': '1299', 'StorageID': '4'}},
                         {'source_row': 4, 'values': {'Capacity': '1259', 'StorageID': '5'}},
                         {'source_row': 5, 'values': {'Capacity': '543', 'StorageID': '6'}},
                         {'source_row': 6, 'values': {'Capacity': '1831', 'StorageID': '7'}},
                         {'source_row': 7, 'values': {'Capacity': '855', 'StorageID': '8'}},
                         {'source_row': 8, 'values': {'Capacity': '619', 'StorageID': '9'}},
                         {'source_row': 9, 'values': {'Capacity': '637', 'StorageID': '10'}},
                         {'source_row': 10, 'values': {'Capacity': '935', 'StorageID': '11'}},
                         {'source_row': 11, 'values': {'Capacity': '626', 'StorageID': '12'}},
                         {'source_row': 12, 'values': {'Capacity': '1457', 'StorageID': '13'}},
                         {'source_row': 13, 'values': {'Capacity': '1198', 'StorageID': '14'}},
                         {'source_row': 14, 'values': {'Capacity': '837', 'StorageID': '15'}}],
             'returned_rows': 15,
             'role': 'storage area capacity',
             'table_id': 'file_0_view_0'},
            {'columns': ['ProductName', 'Value', 'Weight'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {},
             'original_rows': 10,
             'records': [{'source_row': 0, 'values': {'ProductName': 'Window Unit', 'Value': '4811', 'Weight': '114'}},
                         {'source_row': 1,
                          'values': {'ProductName': 'Portable Unit', 'Value': '1130', 'Weight': '200'}},
                         {'source_row': 2, 'values': {'ProductName': 'Split System', 'Value': '1611', 'Weight': '106'}},
                         {'source_row': 3,
                          'values': {'ProductName': 'Ductless System', 'Value': '3368', 'Weight': '256'}},
                         {'source_row': 4, 'values': {'ProductName': 'Central AC', 'Value': '2135', 'Weight': '268'}},
                         {'source_row': 5, 'values': {'ProductName': 'Hybrid AC', 'Value': '1046', 'Weight': '185'}},
                         {'source_row': 6,
                          'values': {'ProductName': 'Geothermal AC', 'Value': '4030', 'Weight': '299'}},
                         {'source_row': 7, 'values': {'ProductName': 'Smart AC', 'Value': '3761', 'Weight': '131'}},
                         {'source_row': 8,
                          'values': {'ProductName': 'Evaporative Cooler', 'Value': '3523', 'Weight': '139'}},
                         {'source_row': 9,
                          'values': {'ProductName': 'Package Unit', 'Value': '1701', 'Weight': '105'}}],
             'returned_rows': 10,
             'role': 'air conditioner product parameters',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB
capacity_table = None
products_table = None
for t in CSVQA_DATA['tables']:
    if t['table_id'] == 'file_0_view_0':
        capacity_table = t
    elif t['table_id'] == 'file_1_view_0':
        products_table = t
if capacity_table is None or products_table is None:
    raise ValueError('Required tables not found in CSVQA_DATA.')
I = []
C = {}
for rec in capacity_table['records']:
    i = rec['values']['StorageID']
    I.append(i)
    try:
        C[i] = int(rec['values']['Capacity'])
    except Exception:
        raise ValueError(f'Invalid capacity for storage area {i}')
J = []
v = {}
w = {}
for rec in products_table['records']:
    j = rec['values']['ProductName']
    J.append(j)
    try:
        v[j] = int(rec['values']['Value'])
        w[j] = int(rec['values']['Weight'])
    except Exception:
        raise ValueError(f'Invalid value or weight for product {j}')
if set(C.keys()) != set(I):
    raise ValueError('Mismatch in storage area indices.')
if set(v.keys()) != set(J) or set(w.keys()) != set(J):
    raise ValueError('Mismatch in product indices.')
m = gp.Model('Amazon_AC_Storage_Allocation')
x_vars = m.addVars(I, J, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((v[j] * x_vars[i, j] for i in I for j in J)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((w[j] * x_vars[i, j] for j in J)) <= C[i] for i in I), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')