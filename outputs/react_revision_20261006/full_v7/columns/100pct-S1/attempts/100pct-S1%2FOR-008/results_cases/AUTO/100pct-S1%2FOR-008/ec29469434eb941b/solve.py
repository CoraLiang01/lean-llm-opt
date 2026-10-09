CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'In the context of New Car Sales in Norway, a car dealership is planning its inventory-replenishment '
          'strategy. For each vehicle type (e.g., sedans, SUVs, electric vehicles, etc.), the dealership has a '
          '‚Äúproducts.csv‚Äù file that records the benefit coefficient for that type. Each vehicle type has a daily '
          'inventory limit, provided in ‚Äúcapacity.csv.‚Äù The objective is to decide how many units of each vehicle '
          'type to order each day so as to maximize total benefit while ensuring that the sum of all ordered units '
          'does not exceed the total inventory capacity. The decision variable x_i represents the number of vehicles '
          'of type i to be ordered per day.The decision variables must be integers.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'columns': ['VehicleType', 'Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0, 'values': {'Capacity': '100', 'VehicleType': 'Sedans'}},
                         {'source_row': 1, 'values': {'Capacity': '80', 'VehicleType': 'SUVs'}},
                         {'source_row': 2, 'values': {'Capacity': '120', 'VehicleType': 'Electric Vehicles'}},
                         {'source_row': 3, 'values': {'Capacity': '90', 'VehicleType': 'Hybrid Vehicles'}},
                         {'source_row': 4, 'values': {'Capacity': '50', 'VehicleType': 'Trucks'}},
                         {'source_row': 5, 'values': {'Capacity': '30', 'VehicleType': 'Sports Cars'}},
                         {'source_row': 6, 'values': {'Capacity': '110', 'VehicleType': 'Compact Cars'}},
                         {'source_row': 7, 'values': {'Capacity': '40', 'VehicleType': 'Luxury Sedans'}},
                         {'source_row': 8, 'values': {'Capacity': '60', 'VehicleType': 'Vans'}},
                         {'source_row': 9, 'values': {'Capacity': '35', 'VehicleType': 'Pickup Trucks'}}],
             'returned_rows': 10,
             'role': 'vehicle type capacity',
             'table_id': 'file_0_view_0'},
            {'columns': ['ProductName', 'Value'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0, 'values': {'ProductName': 'Sedans', 'Value': '1200'}},
                         {'source_row': 1, 'values': {'ProductName': 'SUVs', 'Value': '1800'}},
                         {'source_row': 2, 'values': {'ProductName': 'Electric Vehicles', 'Value': '2500'}},
                         {'source_row': 3, 'values': {'ProductName': 'Hybrid Vehicles', 'Value': '2000'}},
                         {'source_row': 4, 'values': {'ProductName': 'Trucks', 'Value': '1500'}},
                         {'source_row': 5, 'values': {'ProductName': 'Sports Cars', 'Value': '3000'}},
                         {'source_row': 6, 'values': {'ProductName': 'Compact Cars', 'Value': '1000'}},
                         {'source_row': 7, 'values': {'ProductName': 'Luxury Sedans', 'Value': '3500'}},
                         {'source_row': 8, 'values': {'ProductName': 'Vans', 'Value': '1600'}},
                         {'source_row': 9, 'values': {'ProductName': 'Pickup Trucks', 'Value': '1700'}}],
             'returned_rows': 10,
             'role': 'vehicle type benefit coefficient',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem(CSVQA_FRAMES):
    import pandas as pd
    cap_df = CSVQA_FRAMES['file_0_view_0']
    prod_df = CSVQA_FRAMES['file_1_view_0']
    vehicle_types = set(cap_df['VehicleType'].tolist()) | set(prod_df['ProductName'].tolist())
    vehicle_types = list(vehicle_types)
    u_i = {}
    for (_, row) in cap_df.iterrows():
        vt = row['VehicleType']
        cap = row['Capacity']
        try:
            u_i[vt] = int(cap)
        except Exception:
            raise ValueError(f'Invalid Capacity for VehicleType {vt}: {cap}')
    b_i = {}
    for (_, row) in prod_df.iterrows():
        pn = row['ProductName']
        val = row['Value']
        try:
            b_i[pn] = int(val)
        except Exception:
            raise ValueError(f'Invalid Value for ProductName {pn}: {val}')
    for vt in vehicle_types:
        if vt not in u_i:
            raise ValueError(f'Missing Capacity for VehicleType {vt}')
        if vt not in b_i:
            raise ValueError(f'Missing Value for VehicleType {vt}')
    C = sum((u_i[vt] for vt in vehicle_types))
    m = gp.Model('Car_Inventory_Optimization')
    quantity_vars = m.addVars(vehicle_types, lb=0, ub={vt: u_i[vt] for vt in vehicle_types}, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((b_i[vt] * quantity_vars[vt] for vt in vehicle_types)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((quantity_vars[vt] for vt in vehicle_types)) <= C, name='total_capacity')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for vt in vehicle_types:
            print(f'x[{vt}]: {quantity_vars[vt].X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem(CSVQA_FRAMES)