LEGACY_OBSERVATION = '{"values": {"Capacity": "765"}}\n{"values": {"ProductName": "Sedan", "Value": "2524", "Weight": "99"}}\n{"values": {"ProductName": "SUV", "Value": "4614", "Weight": "55"}}\n{"values": {"ProductName": "Truck", "Value": "8416", "Weight": "75"}}\n{"values": {"ProductName": "Convertible", "Value": "5917", "Weight": "94"}}\n{"values": {"ProductName": "Minivan", "Value": "9048", "Weight": "80"}}\n{"values": {"ProductName": "Coupe", "Value": "1140", "Weight": "82"}}\n{"values": {"ProductName": "Hatchback", "Value": "8962", "Weight": "71"}}\n{"values": {"ProductName": "Station Wagon", "Value": "1888", "Weight": "100"}}\n{"values": {"ProductName": "Electric Car", "Value": "8487", "Weight": "28"}}\n{"values": {"ProductName": "Hybrid Car", "Value": "4425", "Weight": "93"}}\n{"values": {"ProductName": "Luxury Sedan", "Value": "4717", "Weight": "84"}}\n{"values": {"ProductName": "Sports Car", "Value": "4210", "Weight": "83"}}\n{"values": {"ProductName": "Crossover", "Value": "1226", "Weight": "62"}}\n{"values": {"ProductName": "Diesel Truck", "Value": "7400", "Weight": "90"}}\n{"values": {"ProductName": "Compact SUV", "Value": "4639", "Weight": "99"}}\n{"values": {"ProductName": "Luxury SUV", "Value": "7712", "Weight": "96"}}\n{"values": {"ProductName": "Cargo Van", "Value": "3299", "Weight": "21"}}\n{"values": {"ProductName": "Pickup Truck", "Value": "9895", "Weight": "39"}}\n{"values": {"ProductName": "Roadster", "Value": "4496", "Weight": "99"}}\n{"values": {"ProductName": "Muscle Car", "Value": "4526", "Weight": "81"}}\n{"values": {"ProductName": "Off-road Vehicle", "Value": "5688", "Weight": "6"}}\n{"values": {"ProductName": "Camper Van", "Value": "3007", "Weight": "58"}}\n{"values": {"ProductName": "Compact Car", "Value": "3623", "Weight": "37"}}\n{"values": {"ProductName": "Motorcycle", "Value": "8474", "Weight": "15"}}\n{"values": {"ProductName": "Electric SUV", "Value": "8372", "Weight": "37"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'Capacity': '765'}}, {'source': '', 'values': {'ProductName': 'Sedan', 'Value': '2524', 'Weight': '99'}}, {'source': '', 'values': {'ProductName': 'SUV', 'Value': '4614', 'Weight': '55'}}, {'source': '', 'values': {'ProductName': 'Truck', 'Value': '8416', 'Weight': '75'}}, {'source': '', 'values': {'ProductName': 'Convertible', 'Value': '5917', 'Weight': '94'}}, {'source': '', 'values': {'ProductName': 'Minivan', 'Value': '9048', 'Weight': '80'}}, {'source': '', 'values': {'ProductName': 'Coupe', 'Value': '1140', 'Weight': '82'}}, {'source': '', 'values': {'ProductName': 'Hatchback', 'Value': '8962', 'Weight': '71'}}, {'source': '', 'values': {'ProductName': 'Station Wagon', 'Value': '1888', 'Weight': '100'}}, {'source': '', 'values': {'ProductName': 'Electric Car', 'Value': '8487', 'Weight': '28'}}, {'source': '', 'values': {'ProductName': 'Hybrid Car', 'Value': '4425', 'Weight': '93'}}, {'source': '', 'values': {'ProductName': 'Luxury Sedan', 'Value': '4717', 'Weight': '84'}}, {'source': '', 'values': {'ProductName': 'Sports Car', 'Value': '4210', 'Weight': '83'}}, {'source': '', 'values': {'ProductName': 'Crossover', 'Value': '1226', 'Weight': '62'}}, {'source': '', 'values': {'ProductName': 'Diesel Truck', 'Value': '7400', 'Weight': '90'}}, {'source': '', 'values': {'ProductName': 'Compact SUV', 'Value': '4639', 'Weight': '99'}}, {'source': '', 'values': {'ProductName': 'Luxury SUV', 'Value': '7712', 'Weight': '96'}}, {'source': '', 'values': {'ProductName': 'Cargo Van', 'Value': '3299', 'Weight': '21'}}, {'source': '', 'values': {'ProductName': 'Pickup Truck', 'Value': '9895', 'Weight': '39'}}, {'source': '', 'values': {'ProductName': 'Roadster', 'Value': '4496', 'Weight': '99'}}, {'source': '', 'values': {'ProductName': 'Muscle Car', 'Value': '4526', 'Weight': '81'}}, {'source': '', 'values': {'ProductName': 'Off-road Vehicle', 'Value': '5688', 'Weight': '6'}}, {'source': '', 'values': {'ProductName': 'Camper Van', 'Value': '3007', 'Weight': '58'}}, {'source': '', 'values': {'ProductName': 'Compact Car', 'Value': '3623', 'Weight': '37'}}, {'source': '', 'values': {'ProductName': 'Motorcycle', 'Value': '8474', 'Weight': '15'}}, {'source': '', 'values': {'ProductName': 'Electric SUV', 'Value': '8372', 'Weight': '37'}}]
from gurobipy import Model, GRB

def solve_vehicle_inventory_optimization():
    global LEGACY_RECORDS
    capacity = None
    products = []
    for rec in LEGACY_RECORDS:
        vals = rec['values']
        if 'Capacity' in vals and vals['Capacity']:
            if capacity is not None:
                raise ValueError('Multiple capacities found in LEGACY_RECORDS')
            capacity = int(vals['Capacity'])
        elif 'ProductName' in vals and vals['ProductName']:
            products.append({'ProductName': vals['ProductName'], 'Value': int(vals['Value']), 'Weight': int(vals['Weight'])})
    if capacity is None:
        raise ValueError('No capacity found in LEGACY_RECORDS')
    if len(products) != 25:
        raise ValueError(f'Expected 25 products, found {len(products)}')
    product_keys = [p['ProductName'] for p in products]
    profit = {p['ProductName']: p['Value'] for p in products}
    weight = {p['ProductName']: p['Weight'] for p in products}
    m = Model()
    m.Params.MIPGap = 0.0001
    x = m.addVars(product_keys, vtype=GRB.INTEGER, lb=0, obj=0, name='')
    m.setObjective(sum((profit[k] * x[k] for k in product_keys)), GRB.MAXIMIZE)
    m.addConstr(sum((weight[k] * x[k] for k in product_keys)) <= capacity, name='')
    if set(profit.keys()) != set(product_keys):
        raise ValueError('Profit keys do not match product keys')
    if set(weight.keys()) != set(product_keys):
        raise ValueError('Weight keys do not match product keys')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for k in product_keys:
            print(f'{x[k].VarName} {x[k].X}')
    else:
        print(f'Solver status: {m.Status}')
m = solve_vehicle_inventory_optimization()