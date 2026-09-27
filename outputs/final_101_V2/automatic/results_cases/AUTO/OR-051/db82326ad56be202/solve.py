LEGACY_OBSERVATION = '{"values": {"CabinetID": "1", "Capacity": "400"}}\n{"values": {"CabinetID": "2", "Capacity": "600"}}\n{"values": {"CabinetID": "3", "Capacity": "500"}}\n{"values": {"CabinetID": "4", "Capacity": "700"}}\n{"values": {"CabinetID": "5", "Capacity": "450"}}\n{"values": {"CabinetID": "6", "Capacity": "650"}}\n{"values": {"CabinetID": "7", "Capacity": "550"}}\n{"values": {"CabinetID": "8", "Capacity": "750"}}\n{"values": {"CabinetID": "9", "Capacity": "480"}}\n{"values": {"CabinetID": "10", "Capacity": "520"}}\n{"values": {"ProductName": "Espresso Beans", "Value": "100", "Weight": "1.0"}}\n{"values": {"ProductName": "Colombian Roast", "Value": "150", "Weight": "1.5"}}\n{"values": {"ProductName": "Arabica Blend", "Value": "80", "Weight": "1.2"}}\n{"values": {"ProductName": "French Roast", "Value": "120", "Weight": "1.3"}}\n{"values": {"ProductName": "Italian Roast", "Value": "130", "Weight": "1.4"}}\n{"values": {"ProductName": "House Blend", "Value": "110", "Weight": "1.1"}}\n{"values": {"ProductName": "Sumatra Coffee", "Value": "160", "Weight": "1.8"}}\n{"values": {"ProductName": "Mocha Java", "Value": "90", "Weight": "1.2"}}\n{"values": {"ProductName": "Hazelnut Flavor", "Value": "95", "Weight": "1.0"}}\n{"values": {"ProductName": "Caramel Blend", "Value": "105", "Weight": "1.3"}}\n{"values": {"ProductName": "Vanilla Flavor", "Value": "85", "Weight": "1.2"}}\n{"values": {"ProductName": "Cappuccino Mix", "Value": "140", "Weight": "1.5"}}\n{"values": {"ProductName": "Pumpkin Spice", "Value": "75", "Weight": "1.1"}}\n{"values": {"ProductName": "Decaf Roast", "Value": "60", "Weight": "1.0"}}\n{"values": {"ProductName": "Organic Roast", "Value": "170", "Weight": "1.6"}}\n{"values": {"ProductName": "Cold Brew", "Value": "115", "Weight": "1.4"}}\n{"values": {"ProductName": "Peruvian Blend", "Value": "155", "Weight": "1.7"}}\n{"values": {"ProductName": "Kenyan AA", "Value": "125", "Weight": "1.3"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'CabinetID': '1', 'Capacity': '400'}}, {'source': '', 'values': {'CabinetID': '2', 'Capacity': '600'}}, {'source': '', 'values': {'CabinetID': '3', 'Capacity': '500'}}, {'source': '', 'values': {'CabinetID': '4', 'Capacity': '700'}}, {'source': '', 'values': {'CabinetID': '5', 'Capacity': '450'}}, {'source': '', 'values': {'CabinetID': '6', 'Capacity': '650'}}, {'source': '', 'values': {'CabinetID': '7', 'Capacity': '550'}}, {'source': '', 'values': {'CabinetID': '8', 'Capacity': '750'}}, {'source': '', 'values': {'CabinetID': '9', 'Capacity': '480'}}, {'source': '', 'values': {'CabinetID': '10', 'Capacity': '520'}}, {'source': '', 'values': {'ProductName': 'Espresso Beans', 'Value': '100', 'Weight': '1.0'}}, {'source': '', 'values': {'ProductName': 'Colombian Roast', 'Value': '150', 'Weight': '1.5'}}, {'source': '', 'values': {'ProductName': 'Arabica Blend', 'Value': '80', 'Weight': '1.2'}}, {'source': '', 'values': {'ProductName': 'French Roast', 'Value': '120', 'Weight': '1.3'}}, {'source': '', 'values': {'ProductName': 'Italian Roast', 'Value': '130', 'Weight': '1.4'}}, {'source': '', 'values': {'ProductName': 'House Blend', 'Value': '110', 'Weight': '1.1'}}, {'source': '', 'values': {'ProductName': 'Sumatra Coffee', 'Value': '160', 'Weight': '1.8'}}, {'source': '', 'values': {'ProductName': 'Mocha Java', 'Value': '90', 'Weight': '1.2'}}, {'source': '', 'values': {'ProductName': 'Hazelnut Flavor', 'Value': '95', 'Weight': '1.0'}}, {'source': '', 'values': {'ProductName': 'Caramel Blend', 'Value': '105', 'Weight': '1.3'}}, {'source': '', 'values': {'ProductName': 'Vanilla Flavor', 'Value': '85', 'Weight': '1.2'}}, {'source': '', 'values': {'ProductName': 'Cappuccino Mix', 'Value': '140', 'Weight': '1.5'}}, {'source': '', 'values': {'ProductName': 'Pumpkin Spice', 'Value': '75', 'Weight': '1.1'}}, {'source': '', 'values': {'ProductName': 'Decaf Roast', 'Value': '60', 'Weight': '1.0'}}, {'source': '', 'values': {'ProductName': 'Organic Roast', 'Value': '170', 'Weight': '1.6'}}, {'source': '', 'values': {'ProductName': 'Cold Brew', 'Value': '115', 'Weight': '1.4'}}, {'source': '', 'values': {'ProductName': 'Peruvian Blend', 'Value': '155', 'Weight': '1.7'}}, {'source': '', 'values': {'ProductName': 'Kenyan AA', 'Value': '125', 'Weight': '1.3'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
cabinets = []
capacity = {}
for rec in records:
    v = rec['values']
    if 'CabinetID' in v and 'Capacity' in v:
        cid = str(v['CabinetID'])
        cabinets.append(cid)
        capacity[cid] = float(v['Capacity'])
products = []
value = {}
weight = {}
for rec in records:
    v = rec['values']
    if 'ProductName' in v and 'Value' in v and ('Weight' in v):
        pname = v['ProductName']
        products.append(pname)
        value[pname] = float(v['Value'])
        weight[pname] = float(v['Weight'])
if len(cabinets) == 0 or len(products) == 0:
    raise ValueError('Missing cabinets or products in LEGACY_RECORDS')
for cid in cabinets:
    if cid not in capacity:
        raise ValueError(f'Missing capacity for cabinet {cid}')
for pname in products:
    if pname not in value or pname not in weight:
        raise ValueError(f'Missing value or weight for product {pname}')
m = gp.Model('CoffeeCabinetAllocation')
x = m.addVars(cabinets, products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value[p] * x[c, p] for c in cabinets for p in products)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weight[p] * x[c, p] for p in products)) <= capacity[c] for c in cabinets), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')