CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A supermarket needs to restock its inventory, and for each type of produce (such as leafy vegetables, '
          'mushrooms, etc.), there is an associated benefit table provided in "products.csv." The supermarket faces an '
          'overall inventory-capacity constraint, provided in “capacity.csv.”. The goal is to decide the daily order '
          'quantity of each produce type so as to maximize total benefit while ensuring that the total weight of all '
          'ordered units does not exceed the overall capacity.The decision variables x_i represents the number of '
          'units of type i of produce to be ordered daily.The decision variables must be integers.',
 'relationships': [],
 'route': 'FLP',
 'tables': [{'columns': ['Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {},
             'original_rows': 1,
             'records': [{'source_row': 0, 'values': {'Capacity': '1035'}}],
             'returned_rows': 1,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'},
            {'columns': ['ProductName', 'Weight', 'Value'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {},
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
    tables = CSVQA_DATA['tables']
    cap_table = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    if not cap_table['records']:
        raise ValueError('Capacity table has no records')
    cap_record = cap_table['records'][0]['values']
    if 'Capacity' not in cap_record:
        raise ValueError("Missing 'Capacity' in capacity record")
    C = float(cap_record['Capacity'])
    prod_table = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
    if not prod_table['records']:
        raise ValueError('Products table has no records')
    I = []
    w = {}
    v = {}
    for rec in prod_table['records']:
        vals = rec['values']
        if 'ProductName' not in vals or 'Weight' not in vals or 'Value' not in vals:
            raise ValueError(f'Missing fields in product record {rec['source_row']}')
        prod = vals['ProductName']
        I.append(prod)
        w[prod] = float(vals['Weight'])
        v[prod] = float(vals['Value'])
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(I, vtype=GRB.INTEGER, lb=0, name='')
    model.setObjective(gurobipy.quicksum((v[i] * x[i] for i in I)), GRB.MAXIMIZE)
    model.addConstr(gurobipy.quicksum((w[i] * x[i] for i in I)) <= C)
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(f'ObjVal: {model.ObjVal}')
        for var in model.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {model.Status}')
    return model
m = solve_problem()