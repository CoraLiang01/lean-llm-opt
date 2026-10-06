LEGACY_OBSERVATION = 'capacity.csv\nCapacity\n1576\n\nproducts.csv\nProductName,Value,Weight\nSedan,1752,15\nSUV,1856,87\nTruck,8372,36\nConvertible,6168,30\nMinivan,9681,33\nCoupe,8062,72\nHatchback,3895,75\nStation Wagon,3254,71\nElectric Car,1701,51\nHybrid Car,6799,21\nLuxury Sedan,2724,97\nSports Car,6304,52\nCrossover,3255,25\nDiesel Truck,1923,15\nCompact SUV,4103,54\nLuxury SUV,4429,57\nCargo Van,2663,18\nPickup Truck,1691,69\nRoadster,5632,26\nMuscle Car,4793,38\nOff-road Vehicle,1343,31\nCamper Van,9124,74\nCompact Car,3652,82\nMotorcycle,8842,49\nElectric SUV,9176,64'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'Capacity': '1576'}}, {'source': 'products.csv', 'values': {'ProductName': 'Sedan', 'Value': '1752', 'Weight': '15'}}, {'source': 'products.csv', 'values': {'ProductName': 'SUV', 'Value': '1856', 'Weight': '87'}}, {'source': 'products.csv', 'values': {'ProductName': 'Truck', 'Value': '8372', 'Weight': '36'}}, {'source': 'products.csv', 'values': {'ProductName': 'Convertible', 'Value': '6168', 'Weight': '30'}}, {'source': 'products.csv', 'values': {'ProductName': 'Minivan', 'Value': '9681', 'Weight': '33'}}, {'source': 'products.csv', 'values': {'ProductName': 'Coupe', 'Value': '8062', 'Weight': '72'}}, {'source': 'products.csv', 'values': {'ProductName': 'Hatchback', 'Value': '3895', 'Weight': '75'}}, {'source': 'products.csv', 'values': {'ProductName': 'Station Wagon', 'Value': '3254', 'Weight': '71'}}, {'source': 'products.csv', 'values': {'ProductName': 'Electric Car', 'Value': '1701', 'Weight': '51'}}, {'source': 'products.csv', 'values': {'ProductName': 'Hybrid Car', 'Value': '6799', 'Weight': '21'}}, {'source': 'products.csv', 'values': {'ProductName': 'Luxury Sedan', 'Value': '2724', 'Weight': '97'}}, {'source': 'products.csv', 'values': {'ProductName': 'Sports Car', 'Value': '6304', 'Weight': '52'}}, {'source': 'products.csv', 'values': {'ProductName': 'Crossover', 'Value': '3255', 'Weight': '25'}}, {'source': 'products.csv', 'values': {'ProductName': 'Diesel Truck', 'Value': '1923', 'Weight': '15'}}, {'source': 'products.csv', 'values': {'ProductName': 'Compact SUV', 'Value': '4103', 'Weight': '54'}}, {'source': 'products.csv', 'values': {'ProductName': 'Luxury SUV', 'Value': '4429', 'Weight': '57'}}, {'source': 'products.csv', 'values': {'ProductName': 'Cargo Van', 'Value': '2663', 'Weight': '18'}}, {'source': 'products.csv', 'values': {'ProductName': 'Pickup Truck', 'Value': '1691', 'Weight': '69'}}, {'source': 'products.csv', 'values': {'ProductName': 'Roadster', 'Value': '5632', 'Weight': '26'}}, {'source': 'products.csv', 'values': {'ProductName': 'Muscle Car', 'Value': '4793', 'Weight': '38'}}, {'source': 'products.csv', 'values': {'ProductName': 'Off-road Vehicle', 'Value': '1343', 'Weight': '31'}}, {'source': 'products.csv', 'values': {'ProductName': 'Camper Van', 'Value': '9124', 'Weight': '74'}}, {'source': 'products.csv', 'values': {'ProductName': 'Compact Car', 'Value': '3652', 'Weight': '82'}}, {'source': 'products.csv', 'values': {'ProductName': 'Motorcycle', 'Value': '8842', 'Weight': '49'}}, {'source': 'products.csv', 'values': {'ProductName': 'Electric SUV', 'Value': '9176', 'Weight': '64'}}]
import gurobipy as gp
from gurobipy import GRB
capacity = None
products = []
values = {}
weights = {}
for rec in LEGACY_RECORDS:
    if rec['source'] == 'capacity.csv':
        if capacity is not None:
            raise ValueError('Multiple capacities found.')
        capacity = int(rec['values']['Capacity'])
    elif rec['source'] == 'products.csv':
        pname = rec['values']['ProductName']
        products.append(pname)
        values[pname] = int(rec['values']['Value'])
        weights[pname] = int(rec['values']['Weight'])
if capacity is None:
    raise ValueError('No capacity found in LEGACY_RECORDS.')
if set(values.keys()) != set(products) or set(weights.keys()) != set(products):
    raise ValueError('Mismatch in product identifiers.')
m = gp.Model('vehicle_inventory_optimization')
x = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((values[i] * x[i] for i in products)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weights[i] * x[i] for i in products)) <= capacity, name='capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')