LEGACY_OBSERVATION = 'capacity.csv\nWarehouse ID,Capacity\nWarehouse 1,100\nWarehouse 2,80\nWarehouse 3,120\nWarehouse 4,90\nWarehouse 5,50\nWarehouse 6,30\nWarehouse 7,110\nWarehouse 8,40\nWarehouse 9,60\nWarehouse 10,35\n\nproducts.csv\nProductName,Value,Weight\nSedans,1200,20\nSUVs,1800,15\nElectric Vehicles,2500,25\nHybrid Vehicles,2000,18\nTrucks,1500,10\nSports Cars,3000,5\nCompact Cars,1000,22\nLuxury Sedans,3500,8\nVans,1600,12\nPickup Trucks,1700,7'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'Warehouse ID': 'Warehouse 1', 'Capacity': '100'}}, {'source': 'capacity.csv', 'values': {'Warehouse ID': 'Warehouse 2', 'Capacity': '80'}}, {'source': 'capacity.csv', 'values': {'Warehouse ID': 'Warehouse 3', 'Capacity': '120'}}, {'source': 'capacity.csv', 'values': {'Warehouse ID': 'Warehouse 4', 'Capacity': '90'}}, {'source': 'capacity.csv', 'values': {'Warehouse ID': 'Warehouse 5', 'Capacity': '50'}}, {'source': 'capacity.csv', 'values': {'Warehouse ID': 'Warehouse 6', 'Capacity': '30'}}, {'source': 'capacity.csv', 'values': {'Warehouse ID': 'Warehouse 7', 'Capacity': '110'}}, {'source': 'capacity.csv', 'values': {'Warehouse ID': 'Warehouse 8', 'Capacity': '40'}}, {'source': 'capacity.csv', 'values': {'Warehouse ID': 'Warehouse 9', 'Capacity': '60'}}, {'source': 'capacity.csv', 'values': {'Warehouse ID': 'Warehouse 10', 'Capacity': '35'}}, {'source': 'products.csv', 'values': {'ProductName': 'Sedans', 'Value': '1200', 'Weight': '20'}}, {'source': 'products.csv', 'values': {'ProductName': 'SUVs', 'Value': '1800', 'Weight': '15'}}, {'source': 'products.csv', 'values': {'ProductName': 'Electric Vehicles', 'Value': '2500', 'Weight': '25'}}, {'source': 'products.csv', 'values': {'ProductName': 'Hybrid Vehicles', 'Value': '2000', 'Weight': '18'}}, {'source': 'products.csv', 'values': {'ProductName': 'Trucks', 'Value': '1500', 'Weight': '10'}}, {'source': 'products.csv', 'values': {'ProductName': 'Sports Cars', 'Value': '3000', 'Weight': '5'}}, {'source': 'products.csv', 'values': {'ProductName': 'Compact Cars', 'Value': '1000', 'Weight': '22'}}, {'source': 'products.csv', 'values': {'ProductName': 'Luxury Sedans', 'Value': '3500', 'Weight': '8'}}, {'source': 'products.csv', 'values': {'ProductName': 'Vans', 'Value': '1600', 'Weight': '12'}}, {'source': 'products.csv', 'values': {'ProductName': 'Pickup Trucks', 'Value': '1700', 'Weight': '7'}}]
from gurobipy import Model, GRB

def solve_inventory_optimization():
    global LEGACY_RECORDS
    warehouses = []
    capacity_w = {}
    for rec in LEGACY_RECORDS:
        if rec['source'] == 'capacity.csv':
            w = rec['values']['Warehouse ID']
            warehouses.append(w)
            try:
                capacity_w[w] = int(rec['values']['Capacity'])
            except Exception:
                raise ValueError(f'Invalid capacity for warehouse {w}')
    products = []
    value_p = {}
    weight_p = {}
    for rec in LEGACY_RECORDS:
        if rec['source'] == 'products.csv':
            p = rec['values']['ProductName']
            products.append(p)
            try:
                value_p[p] = int(rec['values']['Value'])
                weight_p[p] = int(rec['values']['Weight'])
            except Exception:
                raise ValueError(f'Invalid value/weight for product {p}')
    if len(capacity_w) != len(warehouses):
        raise ValueError('Mismatch in warehouse capacity data')
    if len(value_p) != len(products) or len(weight_p) != len(products):
        raise ValueError('Mismatch in product value/weight data')
    m = Model()
    m.Params.MIPGap = 0.0001
    x = m.addVars(warehouses, products, vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(sum((value_p[p] * x[w, p] for w in warehouses for p in products)), GRB.MAXIMIZE)
    for w in warehouses:
        m.addConstr(sum((weight_p[p] * x[w, p] for p in products)) <= capacity_w[w], name='cap_' + w.replace(' ', '_'))
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print('ObjVal:', m.ObjVal)
        for w in warehouses:
            for p in products:
                var = x[w, p]
                print(f'{var.VarName} {var.X}')
    else:
        print('Solver status:', m.Status)
m = solve_inventory_optimization()