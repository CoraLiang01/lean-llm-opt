LEGACY_OBSERVATION = 'capacity.csv\nWarehouse ID,Capacity\nWarehouse 1,100\nWarehouse 2,80\nWarehouse 3,120\nWarehouse 4,90\nWarehouse 5,50\nWarehouse 6,30\nWarehouse 7,110\nWarehouse 8,40\nWarehouse 9,60\nWarehouse 10,35\n\nproducts.csv\nProductName,Value,Weight\nSedans,1200,20\nSUVs,1800,15\nElectric Vehicles,2500,25\nHybrid Vehicles,2000,18\nTrucks,1500,10\nSports Cars,3000,5\nCompact Cars,1000,22\nLuxury Sedans,3500,8\nVans,1600,12\nPickup Trucks,1700,7'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'Warehouse ID': 'Warehouse 1', 'Capacity': '100'}}, {'source': 'capacity.csv', 'values': {'Warehouse ID': 'Warehouse 2', 'Capacity': '80'}}, {'source': 'capacity.csv', 'values': {'Warehouse ID': 'Warehouse 3', 'Capacity': '120'}}, {'source': 'capacity.csv', 'values': {'Warehouse ID': 'Warehouse 4', 'Capacity': '90'}}, {'source': 'capacity.csv', 'values': {'Warehouse ID': 'Warehouse 5', 'Capacity': '50'}}, {'source': 'capacity.csv', 'values': {'Warehouse ID': 'Warehouse 6', 'Capacity': '30'}}, {'source': 'capacity.csv', 'values': {'Warehouse ID': 'Warehouse 7', 'Capacity': '110'}}, {'source': 'capacity.csv', 'values': {'Warehouse ID': 'Warehouse 8', 'Capacity': '40'}}, {'source': 'capacity.csv', 'values': {'Warehouse ID': 'Warehouse 9', 'Capacity': '60'}}, {'source': 'capacity.csv', 'values': {'Warehouse ID': 'Warehouse 10', 'Capacity': '35'}}, {'source': 'products.csv', 'values': {'ProductName': 'Sedans', 'Value': '1200', 'Weight': '20'}}, {'source': 'products.csv', 'values': {'ProductName': 'SUVs', 'Value': '1800', 'Weight': '15'}}, {'source': 'products.csv', 'values': {'ProductName': 'Electric Vehicles', 'Value': '2500', 'Weight': '25'}}, {'source': 'products.csv', 'values': {'ProductName': 'Hybrid Vehicles', 'Value': '2000', 'Weight': '18'}}, {'source': 'products.csv', 'values': {'ProductName': 'Trucks', 'Value': '1500', 'Weight': '10'}}, {'source': 'products.csv', 'values': {'ProductName': 'Sports Cars', 'Value': '3000', 'Weight': '5'}}, {'source': 'products.csv', 'values': {'ProductName': 'Compact Cars', 'Value': '1000', 'Weight': '22'}}, {'source': 'products.csv', 'values': {'ProductName': 'Luxury Sedans', 'Value': '3500', 'Weight': '8'}}, {'source': 'products.csv', 'values': {'ProductName': 'Vans', 'Value': '1600', 'Weight': '12'}}, {'source': 'products.csv', 'values': {'ProductName': 'Pickup Trucks', 'Value': '1700', 'Weight': '7'}}]
from gurobipy import Model, GRB

def solve_inventory_optimization():
    warehouses = []
    warehouse_caps = {}
    products = []
    product_vals = {}
    product_wgts = {}
    for rec in LEGACY_RECORDS:
        if rec['source'] == 'capacity.csv':
            w = rec['values']['Warehouse ID']
            c = int(rec['values']['Capacity'])
            warehouses.append(w)
            warehouse_caps[w] = c
        elif rec['source'] == 'products.csv':
            p = rec['values']['ProductName']
            v = int(rec['values']['Value'])
            wgt = int(rec['values']['Weight'])
            products.append(p)
            product_vals[p] = v
            product_wgts[p] = wgt
    if len(warehouses) == 0 or len(products) == 0:
        raise RuntimeError('Missing warehouse or product data.')
    for w in warehouses:
        if w not in warehouse_caps:
            raise RuntimeError(f'Missing capacity for warehouse {w}')
    for p in products:
        if p not in product_vals or p not in product_wgts:
            raise RuntimeError(f'Missing value or weight for product {p}')
    m = Model()
    m.Params.MIPGap = 0.0001
    x = m.addVars(warehouses, products, vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(sum((product_vals[p] * x[w, p] for w in warehouses for p in products)), GRB.MAXIMIZE)
    for w in warehouses:
        m.addConstr(sum((product_wgts[p] * x[w, p] for p in products)) <= warehouse_caps[w], name='cap_' + w.replace(' ', '_'))
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print('ObjVal', m.ObjVal)
        for w in warehouses:
            for p in products:
                var = x[w, p]
                print(var.VarName, var.X)
    else:
        print('Solver status:', m.Status)
m = solve_inventory_optimization()