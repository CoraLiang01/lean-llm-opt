CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'In the context of New Car Sales in Norway, a car dealership is planning its inventory-replenishment '
          'strategy.For each vehicle type, the dealership has a “products.csv” file that records the benefit '
          'coefficient for that type, in other word, the value for each car.There are some warehouse that can store '
          'these cars.Each warehosue has a capacity limit, provided in “capacity.csv.”The objective is to decide how '
          'many units of each vehicle type to store in the warehouse so as to maximize total value while ensuring that '
          "each warehouse won't exceed capacity limit.The decision variable x_i represents the number of vehicles of "
          'type i to be ordered per day. The decision variables must be integers.',
 'relationships': [],
 'route': 'RA',
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
             'role': 'warehouse capacity limits',
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
             'role': 'vehicle type benefit coefficients',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    products_frame = CSVQA_FRAMES['file_1_view_0']
    capacity_frame = CSVQA_FRAMES['file_0_view_0']
    I = []
    v = {}
    w = {}
    for (source_row, row) in products_frame.iterrows():
        product = row['ProductName']
        I.append(product)
        v[product] = float(row['Value'])
        w[product] = float(row['Weight'])
    J = []
    C = {}
    for (source_row, row) in capacity_frame.iterrows():
        warehouse = row['Warehouse ID']
        J.append(warehouse)
        C[warehouse] = float(row['Capacity'])
    m = gp.Model('Car_Dealership_Inventory')
    m.Params.MIPGap = 0.0001
    x_vars = m.addVars(I, J, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v[i] * x_vars[i, j] for i in I for j in J)), GRB.MAXIMIZE)
    m.addConstrs((gp.quicksum((w[i] * x_vars[i, j] for i in I)) <= C[j] for j in J), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for variable in m.getVars():
            print(f'{variable.VarName}: {variable.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()