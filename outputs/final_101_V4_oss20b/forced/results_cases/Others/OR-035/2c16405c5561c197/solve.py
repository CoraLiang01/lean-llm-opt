CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A small bakery in South Korea, and each day need to stock up on various types of bread. For each type of '
          'bread, we have an expected profit, which can be found in "products.csv." However, the shop has limited '
          'storage capacity, with details provided in "capacity.csv.".Therefore, we must decide which types of bread '
          'to order each day to maximize our total expected profit while staying within our storage limits. The '
          'decision variables x_i represents the number of units of bread type i to be ordered each day.The decision '
          'variables must be integers.',
 'relationships': [],
 'route': 'Others',
 'tables': [{'columns': ['Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {},
             'original_rows': 1,
             'records': [{'source_row': 0, 'values': {'Capacity': '180'}}],
             'returned_rows': 1,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'},
            {'columns': ['ProductName', 'Value', 'Weight'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {},
             'original_rows': 10,
             'records': [{'source_row': 0, 'values': {'ProductName': 'Baguette', 'Value': '888', 'Weight': '4'}},
                         {'source_row': 1, 'values': {'ProductName': 'Croissant', 'Value': '134', 'Weight': '2'}},
                         {'source_row': 2, 'values': {'ProductName': 'Sourdough', 'Value': '129', 'Weight': '4'}},
                         {'source_row': 3, 'values': {'ProductName': 'Rye Bread', 'Value': '370', 'Weight': '3'}},
                         {'source_row': 4, 'values': {'ProductName': 'Brioche', 'Value': '921', 'Weight': '2'}},
                         {'source_row': 5, 'values': {'ProductName': 'Focaccia', 'Value': '765', 'Weight': '1'}},
                         {'source_row': 6, 'values': {'ProductName': 'Ciabatta', 'Value': '154', 'Weight': '2'}},
                         {'source_row': 7, 'values': {'ProductName': 'Pita', 'Value': '837', 'Weight': '1'}},
                         {'source_row': 8, 'values': {'ProductName': 'Bagel', 'Value': '584', 'Weight': '3'}},
                         {'source_row': 9, 'values': {'ProductName': 'English Muffin', 'Value': '365', 'Weight': '3'}}],
             'returned_rows': 10,
             'role': 'descriptive non-unique role',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    capacity_table = next((t for t in tables if t['table_id'] == 'file_0_view_0'), None)
    if capacity_table is None:
        raise ValueError("Missing capacity table 'file_0_view_0'")
    if not capacity_table['records']:
        raise ValueError('Capacity table has no records')
    capacity_record = capacity_table['records'][0]
    if 'Capacity' not in capacity_record['values']:
        raise ValueError('Capacity field missing in record')
    try:
        capacity = float(capacity_record['values']['Capacity'])
    except ValueError:
        raise ValueError('Capacity value is not numeric')
    products_table = next((t for t in tables if t['table_id'] == 'file_1_view_0'), None)
    if products_table is None:
        raise ValueError("Missing products table 'file_1_view_0'")
    if not products_table['records']:
        raise ValueError('Products table has no records')
    I = []
    profit = {}
    weight = {}
    for rec in products_table['records']:
        vals = rec['values']
        if 'ProductName' not in vals or 'Value' not in vals or 'Weight' not in vals:
            raise ValueError(f'Missing required fields in product record {rec['source_row']}')
        pname = vals['ProductName']
        try:
            profit_val = float(vals['Value'])
            weight_val = float(vals['Weight'])
        except ValueError:
            raise ValueError(f'Non-numeric profit or weight in record {rec['source_row']}')
        I.append(pname)
        profit[pname] = profit_val
        weight[pname] = weight_val
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, vtype=GRB.INTEGER, name='')
    obj_expr = gurobipy.quicksum((profit[i] * x[i] for i in I))
    m.setObjective(obj_expr, GRB.MAXIMIZE)
    cap_expr = gurobipy.quicksum((weight[i] * x[i] for i in I))
    m.addConstr(cap_expr <= capacity, name='capacity')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'Optimal objective value: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName} = {var.X}')
else:
    print(f'Solver status: {m.Status}')