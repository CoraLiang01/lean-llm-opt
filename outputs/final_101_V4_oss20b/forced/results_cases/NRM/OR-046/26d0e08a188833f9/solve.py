CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A supermarket needs to replenish its stock, and the ‘products.csv’ provides the relevant income statement '
          'for each type of produce (e.g. leafy greens, mushrooms, etc.). The supermarket also has an overall stock '
          'capacity constraint detailed in ‘capacity.csv’.The objective is to decide which products to order each day '
          'and the quantities to be ordered in order to maximise the overall benefits while adhering to the overall '
          'stock capacity. The decision variable x_i represents the number of units of the ith product to be ordered '
          'each day.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {},
             'original_rows': 1,
             'records': [{'source_row': 0, 'values': {'Capacity': '875'}}],
             'returned_rows': 1,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'},
            {'columns': ['ProductName', 'Weight', 'Value'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {},
             'original_rows': 10,
             'records': [{'source_row': 0, 'values': {'ProductName': 'Spinach', 'Value': '64', 'Weight': '230'}},
                         {'source_row': 1,
                          'values': {'ProductName': 'Shiitake Mushrooms', 'Value': '75', 'Weight': '637'}},
                         {'source_row': 2, 'values': {'ProductName': 'Apples', 'Value': '68', 'Weight': '773'}},
                         {'source_row': 3, 'values': {'ProductName': 'Carrots', 'Value': '11', 'Weight': '653'}},
                         {'source_row': 4, 'values': {'ProductName': 'Basil', 'Value': '91', 'Weight': '755'}},
                         {'source_row': 5, 'values': {'ProductName': 'Potatoes', 'Value': '31', 'Weight': '670'}},
                         {'source_row': 6, 'values': {'ProductName': 'Green Beans', 'Value': '90', 'Weight': '505'}},
                         {'source_row': 7, 'values': {'ProductName': 'Blueberries', 'Value': '56', 'Weight': '821'}},
                         {'source_row': 8, 'values': {'ProductName': 'Oranges', 'Value': '10', 'Weight': '83'}},
                         {'source_row': 9, 'values': {'ProductName': 'Watermelons', 'Value': '24', 'Weight': '249'}}],
             'returned_rows': 10,
             'role': 'descriptive non-unique role',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    capacity_table = next((t for t in CSVQA_DATA['tables'] if t['table_id'] == 'file_0_view_0'), None)
    if capacity_table is None:
        raise ValueError("Capacity table 'file_0_view_0' not found.")
    if not capacity_table['records']:
        raise ValueError('Capacity table has no records.')
    capacity_value = capacity_table['records'][0]['values'].get('Capacity')
    if capacity_value is None:
        raise ValueError('Capacity value missing.')
    try:
        C = int(capacity_value)
    except ValueError:
        raise ValueError(f"Capacity value '{capacity_value}' is not an integer.")
    product_table = next((t for t in CSVQA_DATA['tables'] if t['table_id'] == 'file_1_view_0'), None)
    if product_table is None:
        raise ValueError("Product table 'file_1_view_0' not found.")
    if not product_table['records']:
        raise ValueError('Product table has no records.')
    product_names = []
    weights = {}
    values = {}
    for rec in product_table['records']:
        vals = rec['values']
        name = vals.get('ProductName')
        if name is None:
            raise ValueError(f'Missing ProductName in record {rec['source_row']}.')
        if name in product_names:
            raise ValueError(f"Duplicate product name '{name}'.")
        weight_str = vals.get('Weight')
        value_str = vals.get('Value')
        if weight_str is None or value_str is None:
            raise ValueError(f"Missing Weight or Value for product '{name}'.")
        try:
            w = int(weight_str)
            v = int(value_str)
        except ValueError:
            raise ValueError(f"Non-integer Weight or Value for product '{name}'.")
        product_names.append(name)
        weights[name] = w
        values[name] = v
    if not product_names:
        raise ValueError('No products found.')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(product_names, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((values[p] * x[p] for p in product_names)), GRB.MAXIMIZE)
    m.addConstr(gurobipy.quicksum((weights[p] * x[p] for p in product_names)) <= C, name='capacity')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()