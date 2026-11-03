CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'In BigMart sales, the shop needs to allocate various types of products to different display shelves. The '
          'capacity of each shelf is provided in “capacity.csv,” while each product’s value and weight are in '
          '“products.csv.”The objective is to maximize the total value of products on all shelves without exceeding '
          'any shelf’s capacity. The decision variable  x_{ij} denotes how many units of product 𝑗 are placed on shelf '
          '𝑖.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'columns': ['ShelfID', 'Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0, 'values': {'Capacity': '750', 'ShelfID': '1'}},
                         {'source_row': 1, 'values': {'Capacity': '820', 'ShelfID': '2'}},
                         {'source_row': 2, 'values': {'Capacity': '570', 'ShelfID': '3'}},
                         {'source_row': 3, 'values': {'Capacity': '800', 'ShelfID': '4'}},
                         {'source_row': 4, 'values': {'Capacity': '550', 'ShelfID': '5'}},
                         {'source_row': 5, 'values': {'Capacity': '900', 'ShelfID': '6'}},
                         {'source_row': 6, 'values': {'Capacity': '650', 'ShelfID': '7'}},
                         {'source_row': 7, 'values': {'Capacity': '800', 'ShelfID': '8'}},
                         {'source_row': 8, 'values': {'Capacity': '850', 'ShelfID': '9'}},
                         {'source_row': 9, 'values': {'Capacity': '900', 'ShelfID': '10'}}],
             'returned_rows': 10,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'},
            {'columns': ['ProductName', 'Value', 'Weight'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 20,
             'records': [{'source_row': 0, 'values': {'ProductName': '1', 'Value': '55', 'Weight': '10'}},
                         {'source_row': 1, 'values': {'ProductName': '2', 'Value': '75', 'Weight': '20'}},
                         {'source_row': 2, 'values': {'ProductName': '3', 'Value': '65', 'Weight': '5'}},
                         {'source_row': 3, 'values': {'ProductName': '4', 'Value': '60', 'Weight': '15'}},
                         {'source_row': 4, 'values': {'ProductName': '5', 'Value': '80', 'Weight': '25'}},
                         {'source_row': 5, 'values': {'ProductName': '6', 'Value': '90', 'Weight': '35'}},
                         {'source_row': 6, 'values': {'ProductName': '7', 'Value': '40', 'Weight': '45'}},
                         {'source_row': 7, 'values': {'ProductName': '8', 'Value': '100', 'Weight': '55'}},
                         {'source_row': 8, 'values': {'ProductName': '9', 'Value': '55', 'Weight': '65'}},
                         {'source_row': 9, 'values': {'ProductName': '10', 'Value': '75', 'Weight': '20'}},
                         {'source_row': 10, 'values': {'ProductName': '11', 'Value': '110', 'Weight': '18'}},
                         {'source_row': 11, 'values': {'ProductName': '12', 'Value': '50', 'Weight': '28'}},
                         {'source_row': 12, 'values': {'ProductName': '13', 'Value': '60', 'Weight': '8'}},
                         {'source_row': 13, 'values': {'ProductName': '14', 'Value': '120', 'Weight': '28'}},
                         {'source_row': 14, 'values': {'ProductName': '15', 'Value': '70', 'Weight': '25'}},
                         {'source_row': 15, 'values': {'ProductName': '16', 'Value': '110', 'Weight': '40'}},
                         {'source_row': 16, 'values': {'ProductName': '17', 'Value': '50', 'Weight': '55'}},
                         {'source_row': 17, 'values': {'ProductName': '18', 'Value': '60', 'Weight': '70'}},
                         {'source_row': 18, 'values': {'ProductName': '19', 'Value': '120', 'Weight': '85'}},
                         {'source_row': 19, 'values': {'ProductName': '20', 'Value': '100', 'Weight': '100'}}],
             'returned_rows': 20,
             'role': 'descriptive non-unique role',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    shelf_table = next((t for t in tables if t['table_id'] == 'file_0_view_0'), None)
    if shelf_table is None:
        raise ValueError("Shelf table 'file_0_view_0' not found in CSVQA_DATA")
    product_table = next((t for t in tables if t['table_id'] == 'file_1_view_0'), None)
    if product_table is None:
        raise ValueError("Product table 'file_1_view_0' not found in CSVQA_DATA")
    capacity = {}
    for rec in shelf_table['records']:
        vals = rec['values']
        shelf_id = vals['ShelfID']
        cap_val = vals['Capacity']
        try:
            capacity[shelf_id] = int(cap_val)
        except ValueError:
            raise ValueError(f"Invalid capacity value '{cap_val}' for shelf '{shelf_id}'")
    value = {}
    weight = {}
    for rec in product_table['records']:
        vals = rec['values']
        prod_name = vals['ProductName']
        val_val = vals['Value']
        wt_val = vals['Weight']
        try:
            value[prod_name] = int(val_val)
        except ValueError:
            raise ValueError(f"Invalid value '{val_val}' for product '{prod_name}'")
        try:
            weight[prod_name] = int(wt_val)
        except ValueError:
            raise ValueError(f"Invalid weight '{wt_val}' for product '{prod_name}'")
    shelves = list(capacity.keys())
    products = list(value.keys())
    for prod in products:
        if prod not in weight:
            raise ValueError(f"Missing weight for product '{prod}'")
    for shelf in shelves:
        if shelf not in capacity:
            raise ValueError(f"Missing capacity for shelf '{shelf}'")
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    var_keys = [(i, j) for i in shelves for j in products]
    x = m.addVars(var_keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((value[j] * x[i, j] for i in shelves for j in products)), GRB.MAXIMIZE)
    m.addConstrs((gurobipy.quicksum((weight[j] * x[i, j] for j in products)) <= capacity[i] for i in shelves), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()