CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'In the context of New Car Sales in Norway, a car dealership is planning its inventory-replenishment '
          'strategy.For each vehicle type, the dealership has a “products.csv” file that records the benefit '
          'coefficient for that type, in other word, the value for each car.There are some warehouse that can store '
          'these cars.Each warehosue has a capacity limit, provided in “capacity.csv.”The objective is to decide how '
          'many units of each vehicle type to store in the warehouse so as to maximize total value while ensuring that '
          "each warehouse won't exceed capacity limit.The decision variable x_i represents the number of vehicles of "
          'type i to be ordered per day. The decision variables must be integers.',
 'relationships': [],
 'route': 'TP',
 'tables': [{'columns': ['Warehouse ID', 'Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0, 'values': {'Capacity': '100', 'Warehouse ID': 'Warehouse 1'}},
                         {'source_row': 1, 'values': {'Capacity': '80', 'Warehouse ID': 'Warehouse 2'}},
                         {'source_row': 2, 'values': {'Capacity': '120', 'Warehouse ID': 'Warehouse 3'}},
                         {'source_row': 3, 'values': {'Capacity': '90', 'Warehouse ID': 'Warehouse 4'}},
                         {'source_row': 4, 'values': {'Capacity': '50', 'Warehouse ID': 'Warehouse 5'}},
                         {'source_row': 5, 'values': {'Capacity': '30', 'Warehouse ID': 'Warehouse 6'}},
                         {'source_row': 6, 'values': {'Capacity': '110', 'Warehouse ID': 'Warehouse 7'}},
                         {'source_row': 7, 'values': {'Capacity': '40', 'Warehouse ID': 'Warehouse 8'}},
                         {'source_row': 8, 'values': {'Capacity': '60', 'Warehouse ID': 'Warehouse 9'}},
                         {'source_row': 9, 'values': {'Capacity': '35', 'Warehouse ID': 'Warehouse 10'}}],
             'returned_rows': 10,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'},
            {'columns': ['ProductName', 'Value', 'Weight'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0, 'values': {'ProductName': 'Sedans', 'Value': '1200', 'Weight': '20'}},
                         {'source_row': 1, 'values': {'ProductName': 'SUVs', 'Value': '1800', 'Weight': '15'}},
                         {'source_row': 2,
                          'values': {'ProductName': 'Electric Vehicles', 'Value': '2500', 'Weight': '25'}},
                         {'source_row': 3,
                          'values': {'ProductName': 'Hybrid Vehicles', 'Value': '2000', 'Weight': '18'}},
                         {'source_row': 4, 'values': {'ProductName': 'Trucks', 'Value': '1500', 'Weight': '10'}},
                         {'source_row': 5, 'values': {'ProductName': 'Sports Cars', 'Value': '3000', 'Weight': '5'}},
                         {'source_row': 6, 'values': {'ProductName': 'Compact Cars', 'Value': '1000', 'Weight': '22'}},
                         {'source_row': 7, 'values': {'ProductName': 'Luxury Sedans', 'Value': '3500', 'Weight': '8'}},
                         {'source_row': 8, 'values': {'ProductName': 'Vans', 'Value': '1600', 'Weight': '12'}},
                         {'source_row': 9, 'values': {'ProductName': 'Pickup Trucks', 'Value': '1700', 'Weight': '7'}}],
             'returned_rows': 10,
             'role': 'descriptive non-unique role',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    capacity_table = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    products_table = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
    warehouses = []
    c_w = {}
    for rec in capacity_table['records']:
        values = rec['values']
        wid = values['Warehouse ID']
        cap_str = values['Capacity']
        if wid in c_w:
            raise ValueError(f'Duplicate warehouse ID {wid}')
        try:
            cap = int(cap_str)
        except ValueError:
            raise ValueError(f'Invalid capacity value for {wid}: {cap_str}')
        warehouses.append(wid)
        c_w[wid] = cap
    products = []
    v_p = {}
    w_p = {}
    for rec in products_table['records']:
        values = rec['values']
        pname = values['ProductName']
        val_str = values['Value']
        wt_str = values['Weight']
        if pname in v_p or pname in w_p:
            raise ValueError(f'Duplicate product name {pname}')
        try:
            val = int(val_str)
            wt = int(wt_str)
        except ValueError:
            raise ValueError(f'Invalid numeric value for product {pname}')
        products.append(pname)
        v_p[pname] = val
        w_p[pname] = wt
    for w in warehouses:
        if w not in c_w:
            raise ValueError(f'Missing capacity for warehouse {w}')
    for p in products:
        if p not in v_p or p not in w_p:
            raise ValueError(f'Missing value or weight for product {p}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    var_keys = [(p, w) for p in products for w in warehouses]
    x = model.addVars(var_keys, vtype=GRB.INTEGER, lb=0, name='')
    model.setObjective(gurobipy.quicksum((v_p[p] * x[p, w] for p in products for w in warehouses)), GRB.MAXIMIZE)
    model.addConstrs((gurobipy.quicksum((w_p[p] * x[p, w] for p in products)) <= c_w[w] for w in warehouses), name='')
    model.optimize()
    return model
m = solve_problem()