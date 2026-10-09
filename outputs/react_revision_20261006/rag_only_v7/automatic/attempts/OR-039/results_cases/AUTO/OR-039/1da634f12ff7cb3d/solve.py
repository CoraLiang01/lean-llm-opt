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
             'role': 'vehicle types and value coefficients',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB
import pandas as pd

def solve_problem(CSVQA_FRAMES):
    warehouses_df = CSVQA_FRAMES['file_0_view_0']
    products_df = CSVQA_FRAMES['file_1_view_0']
    W = list(warehouses_df['Warehouse ID'])
    P = list(products_df['ProductName'])
    cap_w = {}
    for (idx, row) in warehouses_df.iterrows():
        wid = row['Warehouse ID']
        cap = row['Capacity']
        if cap == '' or wid == '':
            raise ValueError(f'Missing warehouse capacity or ID at row {idx}')
        cap_w[wid] = float(cap)
    val_p = {}
    wt_p = {}
    for (idx, row) in products_df.iterrows():
        pid = row['ProductName']
        val = row['Value']
        wt = row['Weight']
        if val == '' or wt == '' or pid == '':
            raise ValueError(f'Missing product value, weight, or name at row {idx}')
        val_p[pid] = float(val)
        wt_p[pid] = float(wt)
    if set(W) != set(cap_w.keys()):
        raise ValueError('Warehouse IDs in index set and capacity parameter do not match')
    if set(P) != set(val_p.keys()) or set(P) != set(wt_p.keys()):
        raise ValueError('Product IDs in index set and value/weight parameters do not match')
    m = gp.Model('car_inventory')
    quantity_vars = m.addVars(P, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((val_p[p] * quantity_vars[p] for p in P)), GRB.MAXIMIZE)
    for w in W:
        m.addConstr(gp.quicksum((wt_p[p] * quantity_vars[p] for p in P)) <= cap_w[w], name='cap_' + str(w))
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print('ObjVal:', m.ObjVal)
        for v in quantity_vars.values():
            print(v.VarName, v.X)
    else:
        print('Solver status:', m.Status)
    return m
m = solve_problem(CSVQA_FRAMES)