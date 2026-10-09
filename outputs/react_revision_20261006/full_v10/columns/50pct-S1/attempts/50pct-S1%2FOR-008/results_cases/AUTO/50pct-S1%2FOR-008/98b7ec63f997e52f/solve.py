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
    cap_frame = CSVQA_FRAMES['file_0_view_0']
    veh_types_cap = {}
    for (_, row) in cap_frame.iterrows():
        vt = row['VehicleType']
        cap = float(row['Capacity'])
        veh_types_cap[vt] = cap
    prod_frame = CSVQA_FRAMES['file_1_view_0']
    veh_types_benefit = {}
    for (_, row) in prod_frame.iterrows():
        vt = row['ProductName']
        benefit = float(row['Value'])
        veh_types_benefit[vt] = benefit
    V = sorted(set(veh_types_cap.keys()).union(veh_types_benefit.keys()))
    missing_cap = [i for i in V if i not in veh_types_cap]
    missing_benefit = [i for i in V if i not in veh_types_benefit]
    if missing_cap or missing_benefit:
        raise ValueError(f'Missing capacity for: {missing_cap}, or benefit for: {missing_benefit}')
    b_i = {i: veh_types_benefit[i] for i in V}
    u_i = {i: veh_types_cap[i] for i in V}
    m = gp.Model('Car_Dealership_Inventory')
    m.setParam('MIPGap', 0.0001)
    quantity_vars = m.addVars(V, lb=0, ub=[u_i[i] for i in V], vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((b_i[i] * quantity_vars[i] for i in V)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((quantity_vars[i] for i in V)) <= sum((u_i[i] for i in V)), name='total_inventory_capacity')
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')