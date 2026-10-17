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
 'route': 'FLP',
 'tables': [{'columns': ['Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {},
             'original_rows': 1,
             'records': [{'source_row': 0, 'values': {'Capacity': '4466'}}],
             'returned_rows': 1,
             'role': 'descriptive non-unique role',
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
             'role': 'descriptive non-unique role',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    capacity_table = next((t for t in CSVQA_DATA['tables'] if t['table_id'] == 'file_0_view_0'), None)
    if capacity_table is None:
        raise ValueError("Capacity table 'file_0_view_0' not found")
    if not capacity_table['records']:
        raise ValueError('Capacity table has no records')
    capacity_str = capacity_table['records'][0]['values'].get('Capacity')
    if capacity_str is None:
        raise ValueError('Capacity value missing')
    try:
        C = int(float(capacity_str))
    except Exception:
        raise ValueError('Capacity value is not numeric')
    product_table = next((t for t in CSVQA_DATA['tables'] if t['table_id'] == 'file_1_view_0'), None)
    if product_table is None:
        raise ValueError("Product table 'file_1_view_0' not found")
    product_names = []
    b_i = {}
    for rec in product_table['records']:
        prod = rec['values'].get('ProductName')
        val_str = rec['values'].get('Value')
        if prod is None or val_str is None:
            raise ValueError('Product record missing required fields')
        try:
            val = int(float(val_str))
        except Exception:
            raise ValueError(f'Benefit value not numeric for product {prod}')
        product_names.append(prod)
        b_i[prod] = val
    m = gurobipy.Model('max_benefit')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(product_names, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((b_i[i] * x[i] for i in product_names)), GRB.MAXIMIZE)
    m.addConstr(gurobipy.quicksum((x[i] for i in product_names)) <= C, name='capacity')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()