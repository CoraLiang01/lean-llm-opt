LEGACY_OBSERVATION = 'capacity.csv\nCapacity\n765\n\nproducts.csv\nProductName,Value,Weight\nSedan,2524,99\nSUV,4614,55\nTruck,8416,75\nConvertible,5917,94\nMinivan,9048,80\nCoupe,1140,82\nHatchback,8962,71\nStation Wagon,1888,100\nElectric Car,8487,28\nHybrid Car,4425,93\nLuxury Sedan,4717,84\nSports Car,4210,83\nCrossover,1226,62\nDiesel Truck,7400,90\nCompact SUV,4639,99\nLuxury SUV,7712,96\nCargo Van,3299,21\nPickup Truck,9895,39\nRoadster,4496,99\nMuscle Car,4526,81\nOff-road Vehicle,5688,6\nCamper Van,3007,58\nCompact Car,3623,37\nMotorcycle,8474,15\nElectric SUV,8372,37'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'Capacity': '765'}}, {'source': 'products.csv', 'values': {'ProductName': 'Sedan', 'Value': '2524', 'Weight': '99'}}, {'source': 'products.csv', 'values': {'ProductName': 'SUV', 'Value': '4614', 'Weight': '55'}}, {'source': 'products.csv', 'values': {'ProductName': 'Truck', 'Value': '8416', 'Weight': '75'}}, {'source': 'products.csv', 'values': {'ProductName': 'Convertible', 'Value': '5917', 'Weight': '94'}}, {'source': 'products.csv', 'values': {'ProductName': 'Minivan', 'Value': '9048', 'Weight': '80'}}, {'source': 'products.csv', 'values': {'ProductName': 'Coupe', 'Value': '1140', 'Weight': '82'}}, {'source': 'products.csv', 'values': {'ProductName': 'Hatchback', 'Value': '8962', 'Weight': '71'}}, {'source': 'products.csv', 'values': {'ProductName': 'Station Wagon', 'Value': '1888', 'Weight': '100'}}, {'source': 'products.csv', 'values': {'ProductName': 'Electric Car', 'Value': '8487', 'Weight': '28'}}, {'source': 'products.csv', 'values': {'ProductName': 'Hybrid Car', 'Value': '4425', 'Weight': '93'}}, {'source': 'products.csv', 'values': {'ProductName': 'Luxury Sedan', 'Value': '4717', 'Weight': '84'}}, {'source': 'products.csv', 'values': {'ProductName': 'Sports Car', 'Value': '4210', 'Weight': '83'}}, {'source': 'products.csv', 'values': {'ProductName': 'Crossover', 'Value': '1226', 'Weight': '62'}}, {'source': 'products.csv', 'values': {'ProductName': 'Diesel Truck', 'Value': '7400', 'Weight': '90'}}, {'source': 'products.csv', 'values': {'ProductName': 'Compact SUV', 'Value': '4639', 'Weight': '99'}}, {'source': 'products.csv', 'values': {'ProductName': 'Luxury SUV', 'Value': '7712', 'Weight': '96'}}, {'source': 'products.csv', 'values': {'ProductName': 'Cargo Van', 'Value': '3299', 'Weight': '21'}}, {'source': 'products.csv', 'values': {'ProductName': 'Pickup Truck', 'Value': '9895', 'Weight': '39'}}, {'source': 'products.csv', 'values': {'ProductName': 'Roadster', 'Value': '4496', 'Weight': '99'}}, {'source': 'products.csv', 'values': {'ProductName': 'Muscle Car', 'Value': '4526', 'Weight': '81'}}, {'source': 'products.csv', 'values': {'ProductName': 'Off-road Vehicle', 'Value': '5688', 'Weight': '6'}}, {'source': 'products.csv', 'values': {'ProductName': 'Camper Van', 'Value': '3007', 'Weight': '58'}}, {'source': 'products.csv', 'values': {'ProductName': 'Compact Car', 'Value': '3623', 'Weight': '37'}}, {'source': 'products.csv', 'values': {'ProductName': 'Motorcycle', 'Value': '8474', 'Weight': '15'}}, {'source': 'products.csv', 'values': {'ProductName': 'Electric SUV', 'Value': '8372', 'Weight': '37'}}]
from gurobipy import Model, GRB

def solve_vehicle_inventory_optimization():
    global LEGACY_RECORDS
    capacity_records = [r for r in LEGACY_RECORDS if r['source'] == 'capacity.csv']
    if not capacity_records or 'Capacity' not in capacity_records[0]['values']:
        raise ValueError('Missing capacity in LEGACY_RECORDS')
    try:
        capacity = int(capacity_records[0]['values']['Capacity'])
    except Exception:
        raise ValueError('Capacity value is not an integer')
    product_records = [r for r in LEGACY_RECORDS if r['source'] == 'products.csv']
    if not product_records:
        raise ValueError('No products found in LEGACY_RECORDS')
    products = []
    profit = {}
    weight = {}
    for rec in product_records:
        vals = rec['values']
        if 'ProductName' not in vals or 'Value' not in vals or 'Weight' not in vals:
            raise ValueError('Missing fields in product record: %s' % rec)
        name = vals['ProductName']
        try:
            profit[name] = int(vals['Value'])
            weight[name] = int(vals['Weight'])
        except Exception:
            raise ValueError('Non-integer Value or Weight for product %s' % name)
        products.append(name)
    if len(set(products)) != len(products):
        raise ValueError('Duplicate ProductName entries in LEGACY_RECORDS')
    for name in products:
        if name not in profit or name not in weight:
            raise ValueError('Missing profit or weight for product %s' % name)
    m = Model()
    m.Params.MIPGap = 0.0001
    x = m.addVars(products, vtype=GRB.INTEGER, lb=0, obj=0, name='')
    m.setObjective(sum((profit[name] * x[name] for name in products)), GRB.MAXIMIZE)
    m.addConstr(sum((weight[name] * x[name] for name in products)) <= capacity, name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print('ObjVal', m.ObjVal)
        for name in products:
            print(x[name].VarName, x[name].X)
    else:
        print('Solver status:', m.Status)
    return m
m = solve_vehicle_inventory_optimization()