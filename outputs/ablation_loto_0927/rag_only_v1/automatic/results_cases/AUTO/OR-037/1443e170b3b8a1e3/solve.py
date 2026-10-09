LEGACY_OBSERVATION = 'capacity.csv\n{"values": {"Capacity": "765"}}\n\nproducts.csv\n{"values": {"ProductName": "Sedan", "Value": "2524", "Weight": "99"}}\n{"values": {"ProductName": "SUV", "Value": "4614", "Weight": "55"}}\n{"values": {"ProductName": "Truck", "Value": "8416", "Weight": "75"}}\n{"values": {"ProductName": "Convertible", "Value": "5917", "Weight": "94"}}\n{"values": {"ProductName": "Minivan", "Value": "9048", "Weight": "80"}}\n{"values": {"ProductName": "Coupe", "Value": "1140", "Weight": "82"}}\n{"values": {"ProductName": "Hatchback", "Value": "8962", "Weight": "71"}}\n{"values": {"ProductName": "Station Wagon", "Value": "1888", "Weight": "100"}}\n{"values": {"ProductName": "Electric Car", "Value": "8487", "Weight": "28"}}\n{"values": {"ProductName": "Hybrid Car", "Value": "4425", "Weight": "93"}}\n{"values": {"ProductName": "Luxury Sedan", "Value": "4717", "Weight": "84"}}\n{"values": {"ProductName": "Sports Car", "Value": "4210", "Weight": "83"}}\n{"values": {"ProductName": "Crossover", "Value": "1226", "Weight": "62"}}\n{"values": {"ProductName": "Diesel Truck", "Value": "7400", "Weight": "90"}}\n{"values": {"ProductName": "Compact SUV", "Value": "4639", "Weight": "99"}}\n{"values": {"ProductName": "Luxury SUV", "Value": "7712", "Weight": "96"}}\n{"values": {"ProductName": "Cargo Van", "Value": "3299", "Weight": "21"}}\n{"values": {"ProductName": "Pickup Truck", "Value": "9895", "Weight": "39"}}\n{"values": {"ProductName": "Roadster", "Value": "4496", "Weight": "99"}}\n{"values": {"ProductName": "Muscle Car", "Value": "4526", "Weight": "81"}}\n{"values": {"ProductName": "Off-road Vehicle", "Value": "5688", "Weight": "6"}}\n{"values": {"ProductName": "Camper Van", "Value": "3007", "Weight": "58"}}\n{"values": {"ProductName": "Compact Car", "Value": "3623", "Weight": "37"}}\n{"values": {"ProductName": "Motorcycle", "Value": "8474", "Weight": "15"}}\n{"values": {"ProductName": "Electric SUV", "Value": "8372", "Weight": "37"}}'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'Capacity': '765'}}, {'source': 'products.csv', 'values': {'ProductName': 'Sedan', 'Value': '2524', 'Weight': '99'}}, {'source': 'products.csv', 'values': {'ProductName': 'SUV', 'Value': '4614', 'Weight': '55'}}, {'source': 'products.csv', 'values': {'ProductName': 'Truck', 'Value': '8416', 'Weight': '75'}}, {'source': 'products.csv', 'values': {'ProductName': 'Convertible', 'Value': '5917', 'Weight': '94'}}, {'source': 'products.csv', 'values': {'ProductName': 'Minivan', 'Value': '9048', 'Weight': '80'}}, {'source': 'products.csv', 'values': {'ProductName': 'Coupe', 'Value': '1140', 'Weight': '82'}}, {'source': 'products.csv', 'values': {'ProductName': 'Hatchback', 'Value': '8962', 'Weight': '71'}}, {'source': 'products.csv', 'values': {'ProductName': 'Station Wagon', 'Value': '1888', 'Weight': '100'}}, {'source': 'products.csv', 'values': {'ProductName': 'Electric Car', 'Value': '8487', 'Weight': '28'}}, {'source': 'products.csv', 'values': {'ProductName': 'Hybrid Car', 'Value': '4425', 'Weight': '93'}}, {'source': 'products.csv', 'values': {'ProductName': 'Luxury Sedan', 'Value': '4717', 'Weight': '84'}}, {'source': 'products.csv', 'values': {'ProductName': 'Sports Car', 'Value': '4210', 'Weight': '83'}}, {'source': 'products.csv', 'values': {'ProductName': 'Crossover', 'Value': '1226', 'Weight': '62'}}, {'source': 'products.csv', 'values': {'ProductName': 'Diesel Truck', 'Value': '7400', 'Weight': '90'}}, {'source': 'products.csv', 'values': {'ProductName': 'Compact SUV', 'Value': '4639', 'Weight': '99'}}, {'source': 'products.csv', 'values': {'ProductName': 'Luxury SUV', 'Value': '7712', 'Weight': '96'}}, {'source': 'products.csv', 'values': {'ProductName': 'Cargo Van', 'Value': '3299', 'Weight': '21'}}, {'source': 'products.csv', 'values': {'ProductName': 'Pickup Truck', 'Value': '9895', 'Weight': '39'}}, {'source': 'products.csv', 'values': {'ProductName': 'Roadster', 'Value': '4496', 'Weight': '99'}}, {'source': 'products.csv', 'values': {'ProductName': 'Muscle Car', 'Value': '4526', 'Weight': '81'}}, {'source': 'products.csv', 'values': {'ProductName': 'Off-road Vehicle', 'Value': '5688', 'Weight': '6'}}, {'source': 'products.csv', 'values': {'ProductName': 'Camper Van', 'Value': '3007', 'Weight': '58'}}, {'source': 'products.csv', 'values': {'ProductName': 'Compact Car', 'Value': '3623', 'Weight': '37'}}, {'source': 'products.csv', 'values': {'ProductName': 'Motorcycle', 'Value': '8474', 'Weight': '15'}}, {'source': 'products.csv', 'values': {'ProductName': 'Electric SUV', 'Value': '8372', 'Weight': '37'}}]
from gurobipy import Model, GRB

def solve_vehicle_inventory_optimization():
    global LEGACY_RECORDS
    capacity_records = [r for r in LEGACY_RECORDS if r['source'] == 'capacity.csv']
    if not capacity_records or 'Capacity' not in capacity_records[0]['values']:
        raise ValueError('Missing capacity in LEGACY_RECORDS')
    total_capacity = int(capacity_records[0]['values']['Capacity'])
    product_records = [r for r in LEGACY_RECORDS if r['source'] == 'products.csv']
    if len(product_records) != 25:
        raise ValueError('Expected 25 products, got %d' % len(product_records))
    vehicle_types = []
    profit = {}
    weight = {}
    for rec in product_records:
        name = rec['values']['ProductName']
        vehicle_types.append(name)
        try:
            profit[name] = int(rec['values']['Value'])
            weight[name] = int(rec['values']['Weight'])
        except Exception as e:
            raise ValueError(f'Invalid Value/Weight for {name}: {e}')
    if set(vehicle_types) != set(profit.keys()) or set(vehicle_types) != set(weight.keys()):
        raise ValueError('Mismatch in vehicle_types, profit, or weight keys')
    m = Model()
    m.Params.MIPGap = 0.0001
    x = m.addVars(vehicle_types, vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(sum((profit[v] * x[v] for v in vehicle_types)), GRB.MAXIMIZE)
    m.addConstr(sum((weight[v] * x[v] for v in vehicle_types)) <= total_capacity, name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print('ObjVal', m.ObjVal)
        for v in vehicle_types:
            print(x[v].VarName, x[v].X)
    else:
        print('Status', m.Status)
    return m
m = solve_vehicle_inventory_optimization()