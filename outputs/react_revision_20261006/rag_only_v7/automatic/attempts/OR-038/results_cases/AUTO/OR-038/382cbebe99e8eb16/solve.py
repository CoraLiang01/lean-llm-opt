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
 'tables': [{'columns': ['VehicleType', 'Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {},
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
             'role': 'vehicle type daily inventory limits',
             'table_id': 'file_0_view_0'},
            {'columns': ['ProductName', 'Value'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {},
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
             'role': 'vehicle type benefit coefficients',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem(CSVQA_FRAMES):
    import pandas as pd
    df_cap = CSVQA_FRAMES['file_0_view_0']
    df_val = CSVQA_FRAMES['file_1_view_0']
    cap_types = df_cap['VehicleType'].astype(str)
    val_types = df_val['ProductName'].astype(str)
    cap_map = {v.casefold(): v for v in cap_types}
    val_map = {v.casefold(): v for v in val_types}
    V_keys = sorted(set(cap_map.keys()) & set(val_map.keys()))
    if len(V_keys) == 0:
        raise ValueError('No matching vehicle types between capacity and value tables.')
    V = [cap_map[k] for k in V_keys]
    cap_v = {}
    val_v = {}
    cap_series = df_cap.set_index('VehicleType')['Capacity']
    val_series = df_val.set_index('ProductName')['Value']
    for v in V:
        if v not in cap_series:
            raise ValueError(f"Missing capacity for vehicle type '{v}'")
        if v not in val_series:
            raise ValueError(f"Missing value for vehicle type '{v}'")
        try:
            cap_v[v] = int(cap_series[v])
        except Exception:
            raise ValueError(f"Invalid capacity value for vehicle type '{v}': {cap_series[v]}")
        try:
            val_v[v] = float(val_series[v])
        except Exception:
            raise ValueError(f"Invalid value coefficient for vehicle type '{v}': {val_series[v]}")
    m = gp.Model('vehicle_inventory_optimization')
    quantity_vars = m.addVars(V, lb=0, ub=[cap_v[v] for v in V], vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((val_v[v] * quantity_vars[v] for v in V)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((quantity_vars[v] for v in V)) <= sum((cap_v[v] for v in V)), name='total_inventory_capacity')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)
if m.Status == GRB.OPTIMAL:
    print(m.ObjVal)
    for v in m.getVars():
        print(v.VarName, v.X)
else:
    print(m.Status)