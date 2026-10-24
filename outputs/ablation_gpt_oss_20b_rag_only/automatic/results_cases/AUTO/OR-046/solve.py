CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A supermarket needs to replenish its stock, and the ‘products.csv’ provides the relevant income statement '
          'for each type of produce (e.g. leafy greens, mushrooms, etc.). The supermarket also has an overall stock '
          'capacity constraint detailed in ‘capacity.csv’.The objective is to decide which products to order each day '
          'and the quantities to be ordered in order to maximise the overall benefits while adhering to the overall '
          'stock capacity. The decision variable x_i represents the number of units of the ith product to be ordered '
          'each day.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'columns': ['Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 1,
             'records': [{'source_row': 0, 'values': {'Capacity': '875'}}],
             'returned_rows': 1,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'},
            {'columns': ['ProductName', 'Weight', 'Value'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {'conditions': [], 'logic': 'and'},
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

def solve_problem():
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    tables = CSVQA_DATA['tables']
    cap_table = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    C = float(cap_table['records'][0]['values']['Capacity'])
    prod_table = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
    product_names = []
    v_p = {}
    w_p = {}
    for rec in prod_table['records']:
        pname = rec['values']['ProductName']
        product_names.append(pname)
        v_p[pname] = float(rec['values']['Value'])
        w_p[pname] = float(rec['values']['Weight'])
    if set(product_names) != set(v_p.keys()) or set(product_names) != set(w_p.keys()):
        raise ValueError('Product data mismatch')
    x = m.addVars(product_names, vtype=gurobipy.GRB.INTEGER, lb=0, name='')
    m.setObjective(gurobipy.quicksum((v_p[p] * x[p] for p in product_names)), gurobipy.GRB.MAXIMIZE)
    m.addConstr(gurobipy.quicksum((w_p[p] * x[p] for p in product_names)) <= C)
    m.optimize()
    return m
m = solve_problem()
if m.Status == gurobipy.GRB.OPTIMAL:
    print(m.ObjVal)
    for var in m.getVars():
        print(var.VarName, var.X)
else:
    print(m.Status)