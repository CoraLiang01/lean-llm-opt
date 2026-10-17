CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A supermarket needs to replenish its stock, and the ‘products.csv’ provides the relevant income statement '
          'for each type of produce (e.g. leafy greens, mushrooms, etc.). The supermarket also has an overall stock '
          'capacity constraint detailed in ‘capacity.csv’.The objective is to decide which products to order each day '
          'and the quantities to be ordered in order to maximise the overall benefits while adhering to the overall '
          'stock capacity. The decision variable x_i represents the number of units of the ith product to be ordered '
          'each day.',
 'relationships': [],
 'route': 'FLP',
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
    capacity = None
    product_names = []
    weights = {}
    values = {}
    for table in CSVQA_DATA['tables']:
        if table['table_id'] == 'file_0_view_0':
            if len(table['records']) != 1:
                raise ValueError('Capacity table must contain exactly one record')
            rec = table['records'][0]
            cap_str = rec['values'].get('Capacity')
            if cap_str is None:
                raise ValueError('Capacity value missing')
            capacity = float(cap_str)
        elif table['table_id'] == 'file_1_view_0':
            for rec in table['records']:
                prod = rec['values'].get('ProductName')
                if prod is None:
                    raise ValueError('ProductName missing in product record')
                weight_str = rec['values'].get('Weight')
                value_str = rec['values'].get('Value')
                if weight_str is None or value_str is None:
                    raise ValueError(f'Missing weight or value for product {prod}')
                weight = float(weight_str)
                value = float(value_str)
                product_names.append(prod)
                weights[prod] = weight
                values[prod] = value
    if capacity is None:
        raise ValueError('Capacity not found in data')
    for prod in product_names:
        if prod not in weights or prod not in values:
            raise ValueError(f'Missing data for product {prod}')
    m = gurobipy.Model('Product_Ordering')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(product_names, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gurobipy.quicksum((values[i] * x[i] for i in product_names)), GRB.MAXIMIZE)
    m.addConstr(gurobipy.quicksum((weights[i] * x[i] for i in product_names)) <= capacity)
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')