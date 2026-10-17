CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'In the context of New Car Sales in Norway, a car dealership is planning its inventory-replenishment '
          'strategy.For each vehicle type, the dealership has a “products.csv” file that records the benefit '
          'coefficient for that type, in other word, the value for each car.There are some warehouse that can store '
          'these cars.Each warehosue has a capacity limit, provided in “capacity.csv.”The objective is to decide how '
          'many units of each vehicle type to store in the warehouse so as to maximize total value while ensuring that '
          "each warehouse won't exceed capacity limit.The decision variable x_i represents the number of vehicles of "
          'type i to be ordered per day. The decision variables must be integers.',
 'relationships': [],
 'route': 'FLP',
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
    capacity_table = None
    product_table = None
    for t in tables:
        if t.get('table_id') == 'file_0_view_0':
            capacity_table = t
        elif t.get('table_id') == 'file_1_view_0':
            product_table = t
    if capacity_table is None:
        raise ValueError("Capacity table 'file_0_view_0' not found in CSVQA_DATA")
    if product_table is None:
        raise ValueError("Product table 'file_1_view_0' not found in CSVQA_DATA")
    cap_w = {}
    for rec in capacity_table['records']:
        values = rec['values']
        warehouse_id = values.get('Warehouse ID')
        capacity_str = values.get('Capacity')
        if warehouse_id is None or capacity_str is None:
            raise ValueError(f'Missing Warehouse ID or Capacity in record {rec}')
        try:
            capacity = int(capacity_str)
        except ValueError:
            raise ValueError(f"Invalid capacity value '{capacity_str}' for warehouse '{warehouse_id}'")
        cap_w[warehouse_id] = capacity
    val_i = {}
    for rec in product_table['records']:
        values = rec['values']
        product_name = values.get('ProductName')
        value_str = values.get('Value')
        if product_name is None or value_str is None:
            raise ValueError(f'Missing ProductName or Value in record {rec}')
        try:
            value = int(value_str)
        except ValueError:
            raise ValueError(f"Invalid value '{value_str}' for product '{product_name}'")
        val_i[product_name] = value
    W = list(cap_w.keys())
    P = list(val_i.keys())
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(i, w) for i in P for w in W], vtype=GRB.INTEGER, lb=0, name='')
    obj = gurobipy.quicksum((val_i[i] * x[i, w] for i in P for w in W))
    m.setObjective(obj, GRB.MAXIMIZE)
    m.addConstrs((gurobipy.quicksum((x[i, w] for i in P)) <= cap_w[w] for w in W), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    return m
m = solve_problem()