CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'In the context of New Car Sales in Norway, a car dealership is planning its inventory-replenishment '
          'strategy.For each vehicle type, the dealership has a “products.csv” file that records the benefit '
          'coefficient for that type, in other word, the value for each car.There are some warehouse that can store '
          'these cars.Each warehosue has a capacity limit, provided in “capacity.csv.”The objective is to decide how '
          'many units of each vehicle type to store in the warehouse so as to maximize total value while ensuring that '
          "each warehouse won't exceed capacity limit.The decision variable x_i represents the number of vehicles of "
          'type i to be ordered per day. The decision variables must be integers.',
 'relationships': [],
 'route': 'Others',
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
    data = CSVQA_DATA
    capacity_table = next((table for table in data['tables'] if table['table_id'] == 'file_0_view_0'))
    C_w = {}
    for record in capacity_table['records']:
        values = record['values']
        warehouse_id = values['Warehouse ID']
        capacity = float(values['Capacity'])
        C_w[warehouse_id] = capacity
    product_table = next((table for table in data['tables'] if table['table_id'] == 'file_1_view_0'))
    v_p = {}
    w_p = {}
    for record in product_table['records']:
        values = record['values']
        product_name = values['ProductName']
        v_p[product_name] = float(values['Value'])
        w_p[product_name] = float(values['Weight'])
    warehouses = list(C_w.keys())
    products = list(v_p.keys())
    for p in products:
        if p not in v_p or p not in w_p:
            raise ValueError(f'Missing parameters for product {p}')
    for w in warehouses:
        if w not in C_w:
            raise ValueError(f'Missing capacity for warehouse {w}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars([(p, w) for p in products for w in warehouses], vtype=GRB.INTEGER, lb=0, name='')
    objective = gurobipy.quicksum((v_p[p] * x[p, w] for p in products for w in warehouses))
    model.setObjective(objective, GRB.MAXIMIZE)
    for w in warehouses:
        model.addConstr(gurobipy.quicksum((w_p[p] * x[p, w] for p in products)) <= C_w[w], name='')
    model.optimize()
    return model
m = solve_problem()