CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'In the context of New Car Sales in Norway, a car dealership is planning its inventory-replenishment '
          'strategy. For each vehicle type (e.g., sedans, SUVs, electric vehicles, etc.), the dealership has a '
          '“products.csv” file that records the benefit coefficient for that type. Each vehicle type has a daily '
          'inventory limit, provided in “capacity.csv.” The objective is to decide how many units of each vehicle type '
          'to order each day so as to maximize total benefit while ensuring that the sum of all ordered units does not '
          'exceed the total inventory capacity. The decision variable x_i represents the number of vehicles of type i '
          'to be ordered per day.The decision variables must be integers.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'columns': ['VehicleID', 'VehicleType', 'Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0, 'values': {'Capacity': '100', 'VehicleID': '1', 'VehicleType': 'Sedans'}},
                         {'source_row': 1, 'values': {'Capacity': '80', 'VehicleID': '2', 'VehicleType': 'SUVs'}},
                         {'source_row': 2,
                          'values': {'Capacity': '120', 'VehicleID': '3', 'VehicleType': 'Electric Vehicles'}},
                         {'source_row': 3,
                          'values': {'Capacity': '90', 'VehicleID': '4', 'VehicleType': 'Hybrid Vehicles'}},
                         {'source_row': 4, 'values': {'Capacity': '50', 'VehicleID': '5', 'VehicleType': 'Trucks'}},
                         {'source_row': 5,
                          'values': {'Capacity': '30', 'VehicleID': '6', 'VehicleType': 'Sports Cars'}},
                         {'source_row': 6,
                          'values': {'Capacity': '110', 'VehicleID': '7', 'VehicleType': 'Compact Cars'}},
                         {'source_row': 7,
                          'values': {'Capacity': '40', 'VehicleID': '8', 'VehicleType': 'Luxury Sedans'}},
                         {'source_row': 8, 'values': {'Capacity': '60', 'VehicleID': '9', 'VehicleType': 'Vans'}},
                         {'source_row': 9,
                          'values': {'Capacity': '35', 'VehicleID': '10', 'VehicleType': 'Pickup Trucks'}}],
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

def solve_problem():
    capacity_frame = CSVQA_FRAMES['file_0_view_0']
    products_frame = CSVQA_FRAMES['file_1_view_0']
    benefit_by_type = {}
    for (_, row) in products_frame.iterrows():
        product_name = row['ProductName']
        try:
            value = float(row['Value'])
        except Exception:
            raise ValueError(f"Non-numeric Value for ProductName {product_name}: {row['Value']}")
        benefit_by_type[product_name] = value
    vehicle_ids = []
    capacity_by_id = {}
    benefit_by_id = {}
    for (_, row) in capacity_frame.iterrows():
        vehicle_id = row['VehicleID']
        vehicle_type = row['VehicleType']
        try:
            capacity = float(row['Capacity'])
        except Exception:
            raise ValueError(f"Non-numeric Capacity for VehicleID {vehicle_id}: {row['Capacity']}")
        if vehicle_type not in benefit_by_type:
            raise ValueError(f'VehicleType {vehicle_type} (VehicleID {vehicle_id}) missing in products.csv')
        benefit = benefit_by_type[vehicle_type]
        vehicle_ids.append(vehicle_id)
        capacity_by_id[vehicle_id] = capacity
        benefit_by_id[vehicle_id] = benefit
    m = gp.Model('vehicle_inventory')
    m.setParam('MIPGap', 0.0001)
    quantity_vars = m.addVars(vehicle_ids, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((benefit_by_id[i] * quantity_vars[i] for i in vehicle_ids)), GRB.MAXIMIZE)
    m.addConstrs((quantity_vars[i] <= capacity_by_id[i] for i in vehicle_ids), name='')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')