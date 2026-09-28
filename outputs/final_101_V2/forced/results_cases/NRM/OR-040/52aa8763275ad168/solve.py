CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A real-estate developer is planning property development in New York City. For each area (e.g., Queens, '
          'Brooklyn, etc.), the developer has a “products.csv” file that records the benefit coefficient for that '
          'area.\n'
          '    The developer also faces a single overall development-capacity constraint, provided in “capacity.csv.” '
          'file. The objective is to decide the daily scale of development in each area so as to maximize total '
          'benefit while ensuring that the sum of all development units does not exceed the overall capacity. The '
          'decision variables  x_i  represent the scale of development in area  i  each day.The decision variables '
          'must be integers.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {},
             'original_rows': 1,
             'records': [{'source_row': 0, 'values': {'Capacity': '4466'}}],
             'returned_rows': 1,
             'role': 'overall capacity constraint',
             'table_id': 'file_0_view_0'},
            {'columns': ['ProductName', 'Value'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {},
             'original_rows': 20,
             'records': [{'source_row': 0, 'values': {'ProductName': 'Queens', 'Value': '443'}},
                         {'source_row': 1, 'values': {'ProductName': 'Brooklyn', 'Value': '522'}},
                         {'source_row': 2, 'values': {'ProductName': 'Manhattan', 'Value': '300'}},
                         {'source_row': 3, 'values': {'ProductName': 'Bronx', 'Value': '767'}},
                         {'source_row': 4, 'values': {'ProductName': 'Staten Island', 'Value': '300'}},
                         {'source_row': 5, 'values': {'ProductName': 'Harlem', 'Value': '309'}},
                         {'source_row': 6, 'values': {'ProductName': 'Upper East Side', 'Value': '598'}},
                         {'source_row': 7, 'values': {'ProductName': 'Lower Manhattan', 'Value': '460'}},
                         {'source_row': 8, 'values': {'ProductName': 'Midtown', 'Value': '318'}},
                         {'source_row': 9, 'values': {'ProductName': 'Long Island City', 'Value': '126'}},
                         {'source_row': 10, 'values': {'ProductName': 'Williamsburg', 'Value': '593'}},
                         {'source_row': 11, 'values': {'ProductName': 'Bushwick', 'Value': '871'}},
                         {'source_row': 12, 'values': {'ProductName': 'Flatbush', 'Value': '858'}},
                         {'source_row': 13, 'values': {'ProductName': 'Greenpoint', 'Value': '321'}},
                         {'source_row': 14, 'values': {'ProductName': 'Park Slope', 'Value': '275'}},
                         {'source_row': 15, 'values': {'ProductName': 'Astoria', 'Value': '700'}},
                         {'source_row': 16, 'values': {'ProductName': 'Jackson Heights', 'Value': '685'}},
                         {'source_row': 17, 'values': {'ProductName': 'Flushing', 'Value': '940'}},
                         {'source_row': 18, 'values': {'ProductName': 'Sunnyside', 'Value': '522'}},
                         {'source_row': 19, 'values': {'ProductName': 'Ditmars', 'Value': '763'}}],
             'returned_rows': 20,
             'role': 'decision entities and benefit coefficients',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB
tables = CSVQA_DATA['tables']
capacity_table = None
products_table = None
for t in tables:
    if t['table_id'] == 'file_0_view_0':
        capacity_table = t
    elif t['table_id'] == 'file_1_view_0':
        products_table = t
if capacity_table is None or products_table is None:
    raise ValueError('Required tables not found in CSVQA_DATA.')
capacity_records = capacity_table['records']
if len(capacity_records) != 1:
    raise ValueError('Expected exactly one record in capacity.csv.')
try:
    C = int(capacity_records[0]['values']['Capacity'])
except Exception:
    raise ValueError('Could not parse Capacity from capacity.csv.')
product_records = products_table['records']
I = []
b = {}
for rec in product_records:
    pname = rec['values']['ProductName']
    try:
        val = int(rec['values']['Value'])
    except Exception:
        raise ValueError(f'Could not parse Value for {pname}.')
    I.append(pname)
    b[pname] = val
if set(b.keys()) != set(I):
    raise ValueError('Mismatch in product identifiers and benefit coefficients.')
m = gp.Model('NYC_RealEstate_Dev')
x = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((b[i] * x[i] for i in I)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((x[i] for i in I)) <= C, name='capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for variable in m.getVars():
        print(f'{variable.VarName}: {variable.X}')
else:
    print(f'Solver status: {m.Status}')