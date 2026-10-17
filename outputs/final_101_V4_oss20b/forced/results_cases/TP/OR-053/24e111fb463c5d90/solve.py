CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'In the context of BigMart Sales, the store needs to allocate various types of products into different '
          'display shelves. Specifically, the store has several shelves, each with a capacity limit provided in '
          '“capacity.csv.” The predefined value and weight of each product can be found in “products.csv.” The '
          'objective is to determine the optimal number of units of each product to place on each shelf to maximize '
          'the total value of the products across all shelves, while ensuring that the total weight of the products on '
          'each shelf does not exceed its capacity. The decision variables x_ij represent the number of units of '
          'product j to be placed on shelf i.The decision variables must be integers.',
 'relationships': [],
 'route': 'TP',
 'tables': [{'columns': ['ShelfID', 'Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {},
             'original_rows': 10,
             'records': [{'source_row': 0, 'values': {'Capacity': '500', 'ShelfID': '1'}},
                         {'source_row': 1, 'values': {'Capacity': '700', 'ShelfID': '2'}},
                         {'source_row': 2, 'values': {'Capacity': '600', 'ShelfID': '3'}},
                         {'source_row': 3, 'values': {'Capacity': '800', 'ShelfID': '4'}},
                         {'source_row': 4, 'values': {'Capacity': '550', 'ShelfID': '5'}},
                         {'source_row': 5, 'values': {'Capacity': '900', 'ShelfID': '6'}},
                         {'source_row': 6, 'values': {'Capacity': '650', 'ShelfID': '7'}},
                         {'source_row': 7, 'values': {'Capacity': '750', 'ShelfID': '8'}},
                         {'source_row': 8, 'values': {'Capacity': '820', 'ShelfID': '9'}},
                         {'source_row': 9, 'values': {'Capacity': '570', 'ShelfID': '10'}}],
             'returned_rows': 10,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'},
            {'columns': ['ProductName', 'Value', 'Weight'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {},
             'original_rows': 20,
             'records': [{'source_row': 0, 'values': {'ProductName': '1', 'Value': '50', 'Weight': '10'}},
                         {'source_row': 1, 'values': {'ProductName': '2', 'Value': '70', 'Weight': '20'}},
                         {'source_row': 2, 'values': {'ProductName': '3', 'Value': '30', 'Weight': '5'}},
                         {'source_row': 3, 'values': {'ProductName': '4', 'Value': '60', 'Weight': '15'}},
                         {'source_row': 4, 'values': {'ProductName': '5', 'Value': '80', 'Weight': '25'}},
                         {'source_row': 5, 'values': {'ProductName': '6', 'Value': '90', 'Weight': '30'}},
                         {'source_row': 6, 'values': {'ProductName': '7', 'Value': '40', 'Weight': '12'}},
                         {'source_row': 7, 'values': {'ProductName': '8', 'Value': '100', 'Weight': '35'}},
                         {'source_row': 8, 'values': {'ProductName': '9', 'Value': '55', 'Weight': '10'}},
                         {'source_row': 9, 'values': {'ProductName': '10', 'Value': '75', 'Weight': '20'}},
                         {'source_row': 10, 'values': {'ProductName': '11', 'Value': '65', 'Weight': '18'}},
                         {'source_row': 11, 'values': {'ProductName': '12', 'Value': '95', 'Weight': '28'}},
                         {'source_row': 12, 'values': {'ProductName': '13', 'Value': '45', 'Weight': '8'}},
                         {'source_row': 13, 'values': {'ProductName': '14', 'Value': '85', 'Weight': '22'}},
                         {'source_row': 14, 'values': {'ProductName': '15', 'Value': '70', 'Weight': '25'}},
                         {'source_row': 15, 'values': {'ProductName': '16', 'Value': '110', 'Weight': '40'}},
                         {'source_row': 16, 'values': {'ProductName': '17', 'Value': '50', 'Weight': '14'}},
                         {'source_row': 17, 'values': {'ProductName': '18', 'Value': '60', 'Weight': '16'}},
                         {'source_row': 18, 'values': {'ProductName': '19', 'Value': '120', 'Weight': '50'}},
                         {'source_row': 19, 'values': {'ProductName': '20', 'Value': '100', 'Weight': '30'}}],
             'returned_rows': 20,
             'role': 'descriptive non-unique role',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    capacity_table = None
    products_table = None
    for table in CSVQA_DATA['tables']:
        if table.get('table_id') == 'file_0_view_0':
            capacity_table = table
        elif table.get('table_id') == 'file_1_view_0':
            products_table = table
    if capacity_table is None or products_table is None:
        raise ValueError('Required tables not found in CSVQA_DATA')
    shelves = []
    cap = {}
    for rec in capacity_table['records']:
        sid = rec['values']['ShelfID']
        try:
            sid_int = int(sid)
        except Exception:
            raise ValueError(f"Invalid ShelfID '{sid}'")
        capacity = rec['values']['Capacity']
        try:
            cap_int = int(capacity)
        except Exception:
            raise ValueError(f"Invalid Capacity '{capacity}' for ShelfID {sid}")
        shelves.append(sid_int)
        cap[sid_int] = cap_int
    products = []
    val = {}
    wt = {}
    for rec in products_table['records']:
        pid = rec['values']['ProductName']
        try:
            pid_int = int(pid)
        except Exception:
            raise ValueError(f"Invalid ProductName '{pid}'")
        value = rec['values']['Value']
        weight = rec['values']['Weight']
        try:
            val_int = int(value)
        except Exception:
            raise ValueError(f"Invalid Value '{value}' for ProductName {pid}")
        try:
            wt_int = int(weight)
        except Exception:
            raise ValueError(f"Invalid Weight '{weight}' for ProductName {pid}")
        products.append(pid_int)
        val[pid_int] = val_int
        wt[pid_int] = wt_int
    for i in shelves:
        if i not in cap:
            raise ValueError(f'Missing capacity for shelf {i}')
    for j in products:
        if j not in val or j not in wt:
            raise ValueError(f'Missing value or weight for product {j}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(i, j) for i in shelves for j in products], vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(gurobipy.quicksum((val[j] * x[i, j] for i in shelves for j in products)), GRB.MAXIMIZE)
    for i in shelves:
        m.addConstr(gurobipy.quicksum((wt[j] * x[i, j] for j in products)) <= cap[i], name=f'cap_{i}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for variable in m.getVars():
            print(f'{variable.VarName}: {variable.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()