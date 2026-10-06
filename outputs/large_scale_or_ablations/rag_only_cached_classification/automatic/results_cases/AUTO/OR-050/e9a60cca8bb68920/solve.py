LEGACY_OBSERVATION = '{"values": {"ShelfID": "1", "Capacity": "5.0"}}\n{"values": {"ShelfID": "2", "Capacity": "7.0"}}\n{"values": {"ShelfID": "3", "Capacity": "6.0"}}\n{"values": {"ShelfID": "4", "Capacity": "8.0"}}\n{"values": {"ShelfID": "5", "Capacity": "5.5"}}\n{"values": {"ShelfID": "6", "Capacity": "9.0"}}\n{"values": {"ShelfID": "7", "Capacity": "6.5"}}\n{"values": {"ShelfID": "8", "Capacity": "7.5"}}\n{"values": {"ShelfID": "9", "Capacity": "8.2"}}\n{"values": {"ShelfID": "10", "Capacity": "5.7"}}\n{"values": {"ProductName": "Smartphone", "Value": "200", "Weight": "1.0"}}\n{"values": {"ProductName": "Laptop", "Value": "1500", "Weight": "5.0"}}\n{"values": {"ProductName": "Headphones", "Value": "100", "Weight": "0.5"}}\n{"values": {"ProductName": "Camera", "Value": "800", "Weight": "2.0"}}\n{"values": {"ProductName": "Smartwatch", "Value": "250", "Weight": "0.3"}}\n{"values": {"ProductName": "Tablet", "Value": "600", "Weight": "1.5"}}\n{"values": {"ProductName": "Bluetooth Speaker", "Value": "150", "Weight": "1.0"}}\n{"values": {"ProductName": "Keyboard", "Value": "80", "Weight": "0.8"}}\n{"values": {"ProductName": "Mouse", "Value": "50", "Weight": "0.2"}}\n{"values": {"ProductName": "Monitor", "Value": "300", "Weight": "3.0"}}\n{"values": {"ProductName": "Printer", "Value": "400", "Weight": "4.0"}}\n{"values": {"ProductName": "External Hard Drive", "Value": "120", "Weight": "0.5"}}\n{"values": {"ProductName": "Router", "Value": "60", "Weight": "0.3"}}\n{"values": {"ProductName": "Power Bank", "Value": "40", "Weight": "0.4"}}\n{"values": {"ProductName": "Memory Card", "Value": "30", "Weight": "0.05"}}\n{"values": {"ProductName": "USB Flash Drive", "Value": "25", "Weight": "0.02"}}\n{"values": {"ProductName": "Smart Home Hub", "Value": "100", "Weight": "0.6"}}\n{"values": {"ProductName": "Gaming Console", "Value": "500", "Weight": "4.0"}}\n{"values": {"ProductName": "Fitness Tracker", "Value": "90", "Weight": "0.2"}}\n{"values": {"ProductName": "E-Reader", "Value": "180", "Weight": "0.5"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'ShelfID': '1', 'Capacity': '5.0'}}, {'source': '', 'values': {'ShelfID': '2', 'Capacity': '7.0'}}, {'source': '', 'values': {'ShelfID': '3', 'Capacity': '6.0'}}, {'source': '', 'values': {'ShelfID': '4', 'Capacity': '8.0'}}, {'source': '', 'values': {'ShelfID': '5', 'Capacity': '5.5'}}, {'source': '', 'values': {'ShelfID': '6', 'Capacity': '9.0'}}, {'source': '', 'values': {'ShelfID': '7', 'Capacity': '6.5'}}, {'source': '', 'values': {'ShelfID': '8', 'Capacity': '7.5'}}, {'source': '', 'values': {'ShelfID': '9', 'Capacity': '8.2'}}, {'source': '', 'values': {'ShelfID': '10', 'Capacity': '5.7'}}, {'source': '', 'values': {'ProductName': 'Smartphone', 'Value': '200', 'Weight': '1.0'}}, {'source': '', 'values': {'ProductName': 'Laptop', 'Value': '1500', 'Weight': '5.0'}}, {'source': '', 'values': {'ProductName': 'Headphones', 'Value': '100', 'Weight': '0.5'}}, {'source': '', 'values': {'ProductName': 'Camera', 'Value': '800', 'Weight': '2.0'}}, {'source': '', 'values': {'ProductName': 'Smartwatch', 'Value': '250', 'Weight': '0.3'}}, {'source': '', 'values': {'ProductName': 'Tablet', 'Value': '600', 'Weight': '1.5'}}, {'source': '', 'values': {'ProductName': 'Bluetooth Speaker', 'Value': '150', 'Weight': '1.0'}}, {'source': '', 'values': {'ProductName': 'Keyboard', 'Value': '80', 'Weight': '0.8'}}, {'source': '', 'values': {'ProductName': 'Mouse', 'Value': '50', 'Weight': '0.2'}}, {'source': '', 'values': {'ProductName': 'Monitor', 'Value': '300', 'Weight': '3.0'}}, {'source': '', 'values': {'ProductName': 'Printer', 'Value': '400', 'Weight': '4.0'}}, {'source': '', 'values': {'ProductName': 'External Hard Drive', 'Value': '120', 'Weight': '0.5'}}, {'source': '', 'values': {'ProductName': 'Router', 'Value': '60', 'Weight': '0.3'}}, {'source': '', 'values': {'ProductName': 'Power Bank', 'Value': '40', 'Weight': '0.4'}}, {'source': '', 'values': {'ProductName': 'Memory Card', 'Value': '30', 'Weight': '0.05'}}, {'source': '', 'values': {'ProductName': 'USB Flash Drive', 'Value': '25', 'Weight': '0.02'}}, {'source': '', 'values': {'ProductName': 'Smart Home Hub', 'Value': '100', 'Weight': '0.6'}}, {'source': '', 'values': {'ProductName': 'Gaming Console', 'Value': '500', 'Weight': '4.0'}}, {'source': '', 'values': {'ProductName': 'Fitness Tracker', 'Value': '90', 'Weight': '0.2'}}, {'source': '', 'values': {'ProductName': 'E-Reader', 'Value': '180', 'Weight': '0.5'}}]
import gurobipy as gp
from gurobipy import GRB

def solve_display_allocation(LEGACY_RECORDS):
    capacity_records = [r for r in LEGACY_RECORDS if 'ShelfID' in r['values']]
    product_records = [r for r in LEGACY_RECORDS if 'ProductName' in r['values']]
    if len(capacity_records) != 10:
        raise ValueError('Expected 10 display capacity records, got %d' % len(capacity_records))
    if len(product_records) != 20:
        raise ValueError('Expected 20 product records, got %d' % len(product_records))
    displays = []
    capacities = {}
    for rec in capacity_records:
        shelfid = int(rec['values']['ShelfID'])
        displays.append(shelfid)
        capacities[shelfid] = float(rec['values']['Capacity'])
    products = []
    values = {}
    weights = {}
    for idx, rec in enumerate(product_records):
        pname = rec['values']['ProductName']
        products.append(pname)
        values[pname] = float(rec['values']['Value'])
        weights[pname] = float(rec['values']['Weight'])
    if products[0] != 'Smartphone':
        raise ValueError("First product must be 'Smartphone', got '%s'" % products[0])
    smartphone_name = products[0]
    m = gp.Model()
    m.Params.MIPGap = 0.0001
    x = m.addVars(displays, products, vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((values[j] * x[i, j] for i in displays for j in products)), GRB.MAXIMIZE)
    for i in displays:
        m.addConstr(gp.quicksum((weights[j] * x[i, j] for j in products)) <= capacities[i], name='')
    m.addConstr(gp.quicksum((x[i, smartphone_name] for i in displays)) >= 5, name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print('ObjVal:', m.ObjVal)
        for i in displays:
            for j in products:
                v = x[i, j]
                print(f'{v.VarName} {v.X}')
    else:
        print('Solver status:', m.Status)
    return m
m = solve_display_allocation(LEGACY_RECORDS)