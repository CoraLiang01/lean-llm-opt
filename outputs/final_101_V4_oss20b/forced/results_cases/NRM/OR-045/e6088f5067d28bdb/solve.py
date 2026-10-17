CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A supermarket needs to restock its inventory, and for each type of produce (such as leafy vegetables, '
          'mushrooms, etc.), there is an associated benefit table provided in "products.csv." The supermarket faces an '
          'overall inventory-capacity constraint, provided in “capacity.csv.”. The goal is to decide the daily order '
          'quantity of each produce type so as to maximize total benefit while ensuring that the total weight of all '
          'ordered units does not exceed the overall capacity.The decision variables x_i represents the number of '
          'units of type i of produce to be ordered daily.The decision variables must be integers.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 1,
             'records': [{'source_row': 0, 'values': {'Capacity': '1035'}}],
             'returned_rows': 1,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'},
            {'columns': ['ProductName', 'Weight', 'Value'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0, 'values': {'ProductName': 'Spinach', 'Value': '49', 'Weight': '282'}},
                         {'source_row': 1,
                          'values': {'ProductName': 'Shiitake Mushrooms', 'Value': '30', 'Weight': '83'}},
                         {'source_row': 2, 'values': {'ProductName': 'Apples', 'Value': '30', 'Weight': '251'}},
                         {'source_row': 3, 'values': {'ProductName': 'Carrots', 'Value': '18', 'Weight': '257'}},
                         {'source_row': 4, 'values': {'ProductName': 'Basil', 'Value': '54', 'Weight': '88'}},
                         {'source_row': 5, 'values': {'ProductName': 'Potatoes', 'Value': '27', 'Weight': '52'}},
                         {'source_row': 6, 'values': {'ProductName': 'Green Beans', 'Value': '91', 'Weight': '198'}},
                         {'source_row': 7, 'values': {'ProductName': 'Blueberries', 'Value': '88', 'Weight': '203'}},
                         {'source_row': 8, 'values': {'ProductName': 'Oranges', 'Value': '78', 'Weight': '87'}},
                         {'source_row': 9, 'values': {'ProductName': 'Watermelons', 'Value': '22', 'Weight': '265'}}],
             'returned_rows': 10,
             'role': 'descriptive non-unique role',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    capacity_table = None
    for table in CSVQA_DATA['tables']:
        if table.get('table_id') == 'file_0_view_0':
            capacity_table = table
            break
    if capacity_table is None:
        raise ValueError("Capacity table 'file_0_view_0' not found.")
    if not capacity_table['records']:
        raise ValueError('Capacity table has no records.')
    try:
        C = int(capacity_table['records'][0]['values']['Capacity'])
    except (KeyError, ValueError, TypeError):
        raise ValueError('Capacity value missing or invalid.')
    product_table = None
    for table in CSVQA_DATA['tables']:
        if table.get('table_id') == 'file_1_view_0':
            product_table = table
            break
    if product_table is None:
        raise ValueError("Product table 'file_1_view_0' not found.")
    if not product_table['records']:
        raise ValueError('Product table has no records.')
    I = []
    w = {}
    v = {}
    for record in product_table['records']:
        try:
            name = record['values']['ProductName']
            weight = int(record['values']['Weight'])
            value = int(record['values']['Value'])
        except (KeyError, ValueError, TypeError):
            raise ValueError(f'Invalid product record: {record}')
        if name in I:
            raise ValueError(f'Duplicate product name detected: {name}')
        I.append(name)
        w[name] = weight
        v[name] = value
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((v[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.addConstr(gurobipy.quicksum((w[i] * x[i] for i in I)) <= C, name='capacity')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for variable in m.getVars():
        print(f'{variable.VarName}: {variable.X}')
else:
    print(f'Solver status: {m.Status}')