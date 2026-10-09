LEGACY_OBSERVATION = '{"values": {"Warehouse ID": "Warehouse 1", "Capacity": "100"}}\n{"values": {"Warehouse ID": "Warehouse 2", "Capacity": "80"}}\n{"values": {"Warehouse ID": "Warehouse 3", "Capacity": "120"}}\n{"values": {"Warehouse ID": "Warehouse 4", "Capacity": "90"}}\n{"values": {"Warehouse ID": "Warehouse 5", "Capacity": "50"}}\n{"values": {"Warehouse ID": "Warehouse 6", "Capacity": "30"}}\n{"values": {"Warehouse ID": "Warehouse 7", "Capacity": "110"}}\n{"values": {"Warehouse ID": "Warehouse 8", "Capacity": "40"}}\n{"values": {"Warehouse ID": "Warehouse 9", "Capacity": "60"}}\n{"values": {"Warehouse ID": "Warehouse 10", "Capacity": "35"}}\n{"values": {"ProductName": "Sedans", "Value": "1200", "Weight": "20"}}\n{"values": {"ProductName": "SUVs", "Value": "1800", "Weight": "15"}}\n{"values": {"ProductName": "Electric Vehicles", "Value": "2500", "Weight": "25"}}\n{"values": {"ProductName": "Hybrid Vehicles", "Value": "2000", "Weight": "18"}}\n{"values": {"ProductName": "Trucks", "Value": "1500", "Weight": "10"}}\n{"values": {"ProductName": "Sports Cars", "Value": "3000", "Weight": "5"}}\n{"values": {"ProductName": "Compact Cars", "Value": "1000", "Weight": "22"}}\n{"values": {"ProductName": "Luxury Sedans", "Value": "3500", "Weight": "8"}}\n{"values": {"ProductName": "Vans", "Value": "1600", "Weight": "12"}}\n{"values": {"ProductName": "Pickup Trucks", "Value": "1700", "Weight": "7"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'Warehouse ID': 'Warehouse 1', 'Capacity': '100'}}, {'source': '', 'values': {'Warehouse ID': 'Warehouse 2', 'Capacity': '80'}}, {'source': '', 'values': {'Warehouse ID': 'Warehouse 3', 'Capacity': '120'}}, {'source': '', 'values': {'Warehouse ID': 'Warehouse 4', 'Capacity': '90'}}, {'source': '', 'values': {'Warehouse ID': 'Warehouse 5', 'Capacity': '50'}}, {'source': '', 'values': {'Warehouse ID': 'Warehouse 6', 'Capacity': '30'}}, {'source': '', 'values': {'Warehouse ID': 'Warehouse 7', 'Capacity': '110'}}, {'source': '', 'values': {'Warehouse ID': 'Warehouse 8', 'Capacity': '40'}}, {'source': '', 'values': {'Warehouse ID': 'Warehouse 9', 'Capacity': '60'}}, {'source': '', 'values': {'Warehouse ID': 'Warehouse 10', 'Capacity': '35'}}, {'source': '', 'values': {'ProductName': 'Sedans', 'Value': '1200', 'Weight': '20'}}, {'source': '', 'values': {'ProductName': 'SUVs', 'Value': '1800', 'Weight': '15'}}, {'source': '', 'values': {'ProductName': 'Electric Vehicles', 'Value': '2500', 'Weight': '25'}}, {'source': '', 'values': {'ProductName': 'Hybrid Vehicles', 'Value': '2000', 'Weight': '18'}}, {'source': '', 'values': {'ProductName': 'Trucks', 'Value': '1500', 'Weight': '10'}}, {'source': '', 'values': {'ProductName': 'Sports Cars', 'Value': '3000', 'Weight': '5'}}, {'source': '', 'values': {'ProductName': 'Compact Cars', 'Value': '1000', 'Weight': '22'}}, {'source': '', 'values': {'ProductName': 'Luxury Sedans', 'Value': '3500', 'Weight': '8'}}, {'source': '', 'values': {'ProductName': 'Vans', 'Value': '1600', 'Weight': '12'}}, {'source': '', 'values': {'ProductName': 'Pickup Trucks', 'Value': '1700', 'Weight': '7'}}]
from gurobipy import Model, GRB

def solve_inventory_optimization():
    warehouses = []
    capacity_w = {}
    products = []
    value_p = {}
    weight_p = {}
    for rec in LEGACY_RECORDS:
        v = rec['values']
        if 'Warehouse ID' in v and 'Capacity' in v:
            w = v['Warehouse ID']
            warehouses.append(w)
            capacity_w[w] = int(v['Capacity'])
        if 'ProductName' in v and 'Value' in v and ('Weight' in v):
            p = v['ProductName']
            products.append(p)
            value_p[p] = int(v['Value'])
            weight_p[p] = int(v['Weight'])
    if len(warehouses) == 0 or len(products) == 0:
        raise ValueError('Missing warehouse or product data.')
    for w in warehouses:
        if w not in capacity_w:
            raise ValueError(f'Missing capacity for warehouse {w}')
    for p in products:
        if p not in value_p or p not in weight_p:
            raise ValueError(f'Missing value or weight for product {p}')
    m = Model()
    m.Params.MIPGap = 0.0001
    x = m.addVars(warehouses, products, vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(sum((value_p[p] * x[w, p] for w in warehouses for p in products)), GRB.MAXIMIZE)
    for w in warehouses:
        m.addConstr(sum((weight_p[p] * x[w, p] for p in products)) <= capacity_w[w], name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for w in warehouses:
            for p in products:
                var = x[w, p]
                print(f'{var.VarName} {var.X}')
    else:
        print(f'Solver status: {m.Status}')
m = solve_inventory_optimization()