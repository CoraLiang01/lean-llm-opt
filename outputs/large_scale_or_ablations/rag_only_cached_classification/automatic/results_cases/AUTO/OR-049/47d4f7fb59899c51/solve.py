LEGACY_OBSERVATION = 'capacity.csv\nShelfID,Capacity\n1,5.0\n2,7.0\n3,6.0\n4,8.0\n5,5.5\n6,9.0\n7,6.5\n8,7.5\n9,8.2\n10,5.7\n\nproducts.csv\nProductName,Value,Weight\nSmartphone,200,1.0\nLaptop,1500,5.0\nHeadphones,100,0.5\nCamera,800,2.0\nSmartwatch,250,0.3\nTablet,600,1.5\nBluetooth Speaker,150,1.0\nKeyboard,80,0.8\nMouse,50,0.2\nMonitor,300,3.0\nPrinter,400,4.0\nExternal Hard Drive,120,0.5\nRouter,60,0.3\nPower Bank,40,0.4\nMemory Card,30,0.05\nUSB Flash Drive,25,0.02\nSmart Home Hub,100,0.6\nGaming Console,500,4.0\nFitness Tracker,90,0.2\nE-Reader,180,0.5'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'ShelfID': '1', 'Capacity': '5.0'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '2', 'Capacity': '7.0'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '3', 'Capacity': '6.0'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '4', 'Capacity': '8.0'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '5', 'Capacity': '5.5'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '6', 'Capacity': '9.0'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '7', 'Capacity': '6.5'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '8', 'Capacity': '7.5'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '9', 'Capacity': '8.2'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '10', 'Capacity': '5.7'}}, {'source': 'products.csv', 'values': {'ProductName': 'Smartphone', 'Value': '200', 'Weight': '1.0'}}, {'source': 'products.csv', 'values': {'ProductName': 'Laptop', 'Value': '1500', 'Weight': '5.0'}}, {'source': 'products.csv', 'values': {'ProductName': 'Headphones', 'Value': '100', 'Weight': '0.5'}}, {'source': 'products.csv', 'values': {'ProductName': 'Camera', 'Value': '800', 'Weight': '2.0'}}, {'source': 'products.csv', 'values': {'ProductName': 'Smartwatch', 'Value': '250', 'Weight': '0.3'}}, {'source': 'products.csv', 'values': {'ProductName': 'Tablet', 'Value': '600', 'Weight': '1.5'}}, {'source': 'products.csv', 'values': {'ProductName': 'Bluetooth Speaker', 'Value': '150', 'Weight': '1.0'}}, {'source': 'products.csv', 'values': {'ProductName': 'Keyboard', 'Value': '80', 'Weight': '0.8'}}, {'source': 'products.csv', 'values': {'ProductName': 'Mouse', 'Value': '50', 'Weight': '0.2'}}, {'source': 'products.csv', 'values': {'ProductName': 'Monitor', 'Value': '300', 'Weight': '3.0'}}, {'source': 'products.csv', 'values': {'ProductName': 'Printer', 'Value': '400', 'Weight': '4.0'}}, {'source': 'products.csv', 'values': {'ProductName': 'External Hard Drive', 'Value': '120', 'Weight': '0.5'}}, {'source': 'products.csv', 'values': {'ProductName': 'Router', 'Value': '60', 'Weight': '0.3'}}, {'source': 'products.csv', 'values': {'ProductName': 'Power Bank', 'Value': '40', 'Weight': '0.4'}}, {'source': 'products.csv', 'values': {'ProductName': 'Memory Card', 'Value': '30', 'Weight': '0.05'}}, {'source': 'products.csv', 'values': {'ProductName': 'USB Flash Drive', 'Value': '25', 'Weight': '0.02'}}, {'source': 'products.csv', 'values': {'ProductName': 'Smart Home Hub', 'Value': '100', 'Weight': '0.6'}}, {'source': 'products.csv', 'values': {'ProductName': 'Gaming Console', 'Value': '500', 'Weight': '4.0'}}, {'source': 'products.csv', 'values': {'ProductName': 'Fitness Tracker', 'Value': '90', 'Weight': '0.2'}}, {'source': 'products.csv', 'values': {'ProductName': 'E-Reader', 'Value': '180', 'Weight': '0.5'}}]
from gurobipy import Model, GRB

def solve_shelf_allocation():
    global LEGACY_RECORDS
    shelves = []
    shelf_caps = {}
    for rec in LEGACY_RECORDS:
        if rec['source'] == 'capacity.csv':
            sid = int(rec['values']['ShelfID'])
            cap = float(rec['values']['Capacity'])
            shelves.append(sid)
            shelf_caps[sid] = cap
    products = []
    prod_vals = {}
    prod_wts = {}
    prod_names = {}
    for idx, rec in enumerate(LEGACY_RECORDS):
        if rec['source'] == 'products.csv':
            j = idx - sum((1 for r in LEGACY_RECORDS[:idx] if r['source'] == 'capacity.csv')) + 1
            pname = rec['values']['ProductName']
            val = float(rec['values']['Value'])
            wt = float(rec['values']['Weight'])
            products.append(j)
            prod_vals[j] = val
            prod_wts[j] = wt
            prod_names[j] = pname
    if len(shelves) == 0 or len(products) == 0:
        raise RuntimeError('Missing shelves or products data.')
    for sid in shelves:
        if sid not in shelf_caps:
            raise RuntimeError(f'Missing capacity for shelf {sid}')
    for j in products:
        if j not in prod_vals or j not in prod_wts:
            raise RuntimeError(f'Missing value/weight for product {j}')
    m = Model()
    m.Params.MIPGap = 0.0001
    x = m.addVars(shelves, products, vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(sum((prod_vals[j] * x[i, j] for i in shelves for j in products)), GRB.MAXIMIZE)
    for i in shelves:
        m.addConstr(sum((prod_wts[j] * x[i, j] for j in products)) <= shelf_caps[i], name=f'cap_{i}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for i in shelves:
            for j in products:
                v = x[i, j]
                print(f'{v.VarName} {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_shelf_allocation()