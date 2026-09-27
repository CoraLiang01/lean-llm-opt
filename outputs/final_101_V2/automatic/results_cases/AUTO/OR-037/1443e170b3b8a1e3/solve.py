LEGACY_OBSERVATION = '{"values": {"Capacity": "765"}}\n\n{"values": {"ProductName": "Sedan", "Value": "2524", "Weight": "99"}}\n\n{"values": {"ProductName": "SUV", "Value": "4614", "Weight": "55"}}\n\n{"values": {"ProductName": "Truck", "Value": "8416", "Weight": "75"}}\n\n{"values": {"ProductName": "Convertible", "Value": "5917", "Weight": "94"}}\n\n{"values": {"ProductName": "Minivan", "Value": "9048", "Weight": "80"}}\n\n{"values": {"ProductName": "Coupe", "Value": "1140", "Weight": "82"}}\n\n{"values": {"ProductName": "Hatchback", "Value": "8962", "Weight": "71"}}\n\n{"values": {"ProductName": "Station Wagon", "Value": "1888", "Weight": "100"}}\n\n{"values": {"ProductName": "Electric Car", "Value": "8487", "Weight": "28"}}\n\n{"values": {"ProductName": "Hybrid Car", "Value": "4425", "Weight": "93"}}\n\n{"values": {"ProductName": "Luxury Sedan", "Value": "4717", "Weight": "84"}}\n\n{"values": {"ProductName": "Sports Car", "Value": "4210", "Weight": "83"}}\n\n{"values": {"ProductName": "Crossover", "Value": "1226", "Weight": "62"}}\n\n{"values": {"ProductName": "Diesel Truck", "Value": "7400", "Weight": "90"}}\n\n{"values": {"ProductName": "Compact SUV", "Value": "4639", "Weight": "99"}}\n\n{"values": {"ProductName": "Luxury SUV", "Value": "7712", "Weight": "96"}}\n\n{"values": {"ProductName": "Cargo Van", "Value": "3299", "Weight": "21"}}\n\n{"values": {"ProductName": "Pickup Truck", "Value": "9895", "Weight": "39"}}\n\n{"values": {"ProductName": "Roadster", "Value": "4496", "Weight": "99"}}\n\n{"values": {"ProductName": "Muscle Car", "Value": "4526", "Weight": "81"}}\n\n{"values": {"ProductName": "Off-road Vehicle", "Value": "5688", "Weight": "6"}}\n\n{"values": {"ProductName": "Camper Van", "Value": "3007", "Weight": "58"}}\n\n{"values": {"ProductName": "Compact Car", "Value": "3623", "Weight": "37"}}\n\n{"values": {"ProductName": "Motorcycle", "Value": "8474", "Weight": "15"}}\n\n{"values": {"ProductName": "Electric SUV", "Value": "8372", "Weight": "37"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'Capacity': '765'}}, {'source': '', 'values': {'ProductName': 'Sedan', 'Value': '2524', 'Weight': '99'}}, {'source': '', 'values': {'ProductName': 'SUV', 'Value': '4614', 'Weight': '55'}}, {'source': '', 'values': {'ProductName': 'Truck', 'Value': '8416', 'Weight': '75'}}, {'source': '', 'values': {'ProductName': 'Convertible', 'Value': '5917', 'Weight': '94'}}, {'source': '', 'values': {'ProductName': 'Minivan', 'Value': '9048', 'Weight': '80'}}, {'source': '', 'values': {'ProductName': 'Coupe', 'Value': '1140', 'Weight': '82'}}, {'source': '', 'values': {'ProductName': 'Hatchback', 'Value': '8962', 'Weight': '71'}}, {'source': '', 'values': {'ProductName': 'Station Wagon', 'Value': '1888', 'Weight': '100'}}, {'source': '', 'values': {'ProductName': 'Electric Car', 'Value': '8487', 'Weight': '28'}}, {'source': '', 'values': {'ProductName': 'Hybrid Car', 'Value': '4425', 'Weight': '93'}}, {'source': '', 'values': {'ProductName': 'Luxury Sedan', 'Value': '4717', 'Weight': '84'}}, {'source': '', 'values': {'ProductName': 'Sports Car', 'Value': '4210', 'Weight': '83'}}, {'source': '', 'values': {'ProductName': 'Crossover', 'Value': '1226', 'Weight': '62'}}, {'source': '', 'values': {'ProductName': 'Diesel Truck', 'Value': '7400', 'Weight': '90'}}, {'source': '', 'values': {'ProductName': 'Compact SUV', 'Value': '4639', 'Weight': '99'}}, {'source': '', 'values': {'ProductName': 'Luxury SUV', 'Value': '7712', 'Weight': '96'}}, {'source': '', 'values': {'ProductName': 'Cargo Van', 'Value': '3299', 'Weight': '21'}}, {'source': '', 'values': {'ProductName': 'Pickup Truck', 'Value': '9895', 'Weight': '39'}}, {'source': '', 'values': {'ProductName': 'Roadster', 'Value': '4496', 'Weight': '99'}}, {'source': '', 'values': {'ProductName': 'Muscle Car', 'Value': '4526', 'Weight': '81'}}, {'source': '', 'values': {'ProductName': 'Off-road Vehicle', 'Value': '5688', 'Weight': '6'}}, {'source': '', 'values': {'ProductName': 'Camper Van', 'Value': '3007', 'Weight': '58'}}, {'source': '', 'values': {'ProductName': 'Compact Car', 'Value': '3623', 'Weight': '37'}}, {'source': '', 'values': {'ProductName': 'Motorcycle', 'Value': '8474', 'Weight': '15'}}, {'source': '', 'values': {'ProductName': 'Electric SUV', 'Value': '8372', 'Weight': '37'}}]
import gurobipy as gp
from gurobipy import GRB
capacity = None
products = []
profit = {}
weight = {}
for rec in LEGACY_RECORDS:
    vals = rec['values']
    if 'Capacity' in vals:
        if capacity is not None:
            raise ValueError('Multiple capacities found')
        capacity = int(vals['Capacity'])
    elif 'ProductName' in vals and 'Value' in vals and ('Weight' in vals):
        pname = vals['ProductName']
        products.append(pname)
        profit[pname] = int(vals['Value'])
        weight[pname] = int(vals['Weight'])
if capacity is None:
    raise ValueError('No capacity found')
if set(profit.keys()) != set(products) or set(weight.keys()) != set(products):
    raise ValueError('Profit/weight data missing for some products')
m = gp.Model('Car_Inventory_Optimization')
x = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((profit[p] * x[p] for p in products)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[p] * x[p] for p in products)) <= capacity, name='capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')