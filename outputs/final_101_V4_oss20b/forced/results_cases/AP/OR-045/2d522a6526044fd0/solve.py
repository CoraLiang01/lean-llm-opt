CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A supermarket needs to restock its inventory, and for each type of produce (such as leafy vegetables, '
          'mushrooms, etc.), there is an associated benefit table provided in "products.csv." The supermarket faces an '
          'overall inventory-capacity constraint, provided in “capacity.csv.”. The goal is to decide the daily order '
          'quantity of each produce type so as to maximize total benefit while ensuring that the total weight of all '
          'ordered units does not exceed the overall capacity.The decision variables x_i represents the number of '
          'units of type i of produce to be ordered daily.The decision variables must be integers.',
 'relationships': [],
 'route': 'AP',
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
    capacity_table = None
    for tbl in CSVQA_DATA['tables']:
        if tbl.get('table_id') == 'file_0_view_0':
            capacity_table = tbl
            break
    if capacity_table is None:
        raise ValueError("Capacity table 'file_0_view_0' not found.")
    if not capacity_table['records']:
        raise ValueError('Capacity table has no records.')
    cap_val_str = capacity_table['records'][0]['values'].get('Capacity')
    if cap_val_str is None:
        raise ValueError('Capacity value missing.')
    try:
        C = int(cap_val_str)
    except ValueError:
        raise ValueError(f"Capacity value '{cap_val_str}' is not an integer.")
    product_table = None
    for tbl in CSVQA_DATA['tables']:
        if tbl.get('table_id') == 'file_1_view_0':
            product_table = tbl
            break
    if product_table is None:
        raise ValueError("Product table 'file_1_view_0' not found.")
    records = product_table.get('records', [])
    if not records:
        raise ValueError('Product table has no records.')
    product_names = []
    weight_dict = {}
    value_dict = {}
    for rec in records:
        vals = rec.get('values', {})
        name = vals.get('ProductName')
        weight_str = vals.get('Weight')
        value_str = vals.get('Value')
        if name is None or weight_str is None or value_str is None:
            raise ValueError(f'Missing data in record {rec.get('source_row')}.')
        try:
            weight = int(weight_str)
            value = int(value_str)
        except ValueError:
            raise ValueError(f'Non-integer weight/value in record {rec.get('source_row')}.')
        if name in product_names:
            raise ValueError(f"Duplicate product name '{name}'.")
        product_names.append(name)
        weight_dict[name] = weight
        value_dict[name] = value
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(product_names, lb=0, vtype=GRB.INTEGER, name='')
    model.setObjective(gurobipy.quicksum((value_dict[i] * x[i] for i in product_names)), GRB.MAXIMIZE)
    model.addConstr(gurobipy.quicksum((weight_dict[i] * x[i] for i in product_names)) <= C)
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(f'ObjVal: {model.ObjVal}')
        for var in model.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {model.Status}')
    return model
m = solve_problem()