LEGACY_OBSERVATION = '{"values": {"BookshelfID": "1", "Capacity": "200"}}\n{"values": {"BookshelfID": "2", "Capacity": "200"}}\n{"values": {"BookshelfID": "3", "Capacity": "300"}}\n{"values": {"BookshelfID": "4", "Capacity": "400"}}\n{"values": {"BookshelfID": "5", "Capacity": "550"}}\n{"values": {"BookshelfID": "6", "Capacity": "600"}}\n{"values": {"BookshelfID": "7", "Capacity": "650"}}\n{"values": {"BookshelfID": "8", "Capacity": "750"}}\n{"values": {"BookshelfID": "9", "Capacity": "820"}}\n{"values": {"BookshelfID": "10", "Capacity": "570"}}\n{"values": {"ProductName": "The Great Gatsby", "Value": "50", "Weight": "10"}}\n{"values": {"ProductName": "To Kill a Mockingbird", "Value": "70", "Weight": "20"}}\n{"values": {"ProductName": "1984", "Value": "30", "Weight": "5"}}\n{"values": {"ProductName": "Pride and Prejudice", "Value": "60", "Weight": "15"}}\n{"values": {"ProductName": "The Catcher in the Rye", "Value": "80", "Weight": "25"}}\n{"values": {"ProductName": "Moby Dick", "Value": "90", "Weight": "30"}}\n{"values": {"ProductName": "Jane Eyre", "Value": "40", "Weight": "12"}}\n{"values": {"ProductName": "War and Peace", "Value": "100", "Weight": "35"}}\n{"values": {"ProductName": "The Odyssey", "Value": "55", "Weight": "10"}}\n{"values": {"ProductName": "Crime and Punishment", "Value": "75", "Weight": "20"}}\n{"values": {"ProductName": "The Hobbit", "Value": "65", "Weight": "18"}}\n{"values": {"ProductName": "Brave New World", "Value": "95", "Weight": "28"}}\n{"values": {"ProductName": "Anna Karenina", "Value": "45", "Weight": "8"}}\n{"values": {"ProductName": "Wuthering Heights", "Value": "85", "Weight": "22"}}\n{"values": {"ProductName": "The Divine Comedy", "Value": "70", "Weight": "25"}}\n{"values": {"ProductName": "The Iliad", "Value": "110", "Weight": "40"}}\n{"values": {"ProductName": "Les Misérables", "Value": "50", "Weight": "14"}}\n{"values": {"ProductName": "Dracula", "Value": "60", "Weight": "16"}}\n{"values": {"ProductName": "Frankenstein", "Value": "120", "Weight": "50"}}\n{"values": {"ProductName": "The Brothers Karamazov", "Value": "100", "Weight": "30"}}\n{"values": {"ProductName": "Don Quixote", "Value": "52", "Weight": "11"}}\n{"values": {"ProductName": "One Hundred Years of Solitude", "Value": "68", "Weight": "19"}}\n{"values": {"ProductName": "Ulysses", "Value": "38", "Weight": "7"}}\n{"values": {"ProductName": "The Alchemist", "Value": "58", "Weight": "14"}}\n{"values": {"ProductName": "Meditations", "Value": "82", "Weight": "24"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'BookshelfID': '1', 'Capacity': '200'}}, {'source': '', 'values': {'BookshelfID': '2', 'Capacity': '200'}}, {'source': '', 'values': {'BookshelfID': '3', 'Capacity': '300'}}, {'source': '', 'values': {'BookshelfID': '4', 'Capacity': '400'}}, {'source': '', 'values': {'BookshelfID': '5', 'Capacity': '550'}}, {'source': '', 'values': {'BookshelfID': '6', 'Capacity': '600'}}, {'source': '', 'values': {'BookshelfID': '7', 'Capacity': '650'}}, {'source': '', 'values': {'BookshelfID': '8', 'Capacity': '750'}}, {'source': '', 'values': {'BookshelfID': '9', 'Capacity': '820'}}, {'source': '', 'values': {'BookshelfID': '10', 'Capacity': '570'}}, {'source': '', 'values': {'ProductName': 'The Great Gatsby', 'Value': '50', 'Weight': '10'}}, {'source': '', 'values': {'ProductName': 'To Kill a Mockingbird', 'Value': '70', 'Weight': '20'}}, {'source': '', 'values': {'ProductName': '1984', 'Value': '30', 'Weight': '5'}}, {'source': '', 'values': {'ProductName': 'Pride and Prejudice', 'Value': '60', 'Weight': '15'}}, {'source': '', 'values': {'ProductName': 'The Catcher in the Rye', 'Value': '80', 'Weight': '25'}}, {'source': '', 'values': {'ProductName': 'Moby Dick', 'Value': '90', 'Weight': '30'}}, {'source': '', 'values': {'ProductName': 'Jane Eyre', 'Value': '40', 'Weight': '12'}}, {'source': '', 'values': {'ProductName': 'War and Peace', 'Value': '100', 'Weight': '35'}}, {'source': '', 'values': {'ProductName': 'The Odyssey', 'Value': '55', 'Weight': '10'}}, {'source': '', 'values': {'ProductName': 'Crime and Punishment', 'Value': '75', 'Weight': '20'}}, {'source': '', 'values': {'ProductName': 'The Hobbit', 'Value': '65', 'Weight': '18'}}, {'source': '', 'values': {'ProductName': 'Brave New World', 'Value': '95', 'Weight': '28'}}, {'source': '', 'values': {'ProductName': 'Anna Karenina', 'Value': '45', 'Weight': '8'}}, {'source': '', 'values': {'ProductName': 'Wuthering Heights', 'Value': '85', 'Weight': '22'}}, {'source': '', 'values': {'ProductName': 'The Divine Comedy', 'Value': '70', 'Weight': '25'}}, {'source': '', 'values': {'ProductName': 'The Iliad', 'Value': '110', 'Weight': '40'}}, {'source': '', 'values': {'ProductName': 'Les Misérables', 'Value': '50', 'Weight': '14'}}, {'source': '', 'values': {'ProductName': 'Dracula', 'Value': '60', 'Weight': '16'}}, {'source': '', 'values': {'ProductName': 'Frankenstein', 'Value': '120', 'Weight': '50'}}, {'source': '', 'values': {'ProductName': 'The Brothers Karamazov', 'Value': '100', 'Weight': '30'}}, {'source': '', 'values': {'ProductName': 'Don Quixote', 'Value': '52', 'Weight': '11'}}, {'source': '', 'values': {'ProductName': 'One Hundred Years of Solitude', 'Value': '68', 'Weight': '19'}}, {'source': '', 'values': {'ProductName': 'Ulysses', 'Value': '38', 'Weight': '7'}}, {'source': '', 'values': {'ProductName': 'The Alchemist', 'Value': '58', 'Weight': '14'}}, {'source': '', 'values': {'ProductName': 'Meditations', 'Value': '82', 'Weight': '24'}}]
import gurobipy as gp
from gurobipy import GRB
bookshelf_records = [r for r in LEGACY_RECORDS if 'BookshelfID' in r['values']]
product_records = [r for r in LEGACY_RECORDS if 'ProductName' in r['values']]
bookshelves = []
capacity = {}
for rec in bookshelf_records:
    i = rec['values']['BookshelfID']
    c = rec['values']['Capacity']
    bookshelves.append(i)
    capacity[i] = int(c)
products = []
value = {}
weight = {}
for rec in product_records:
    j = rec['values']['ProductName']
    v = rec['values']['Value']
    w = rec['values']['Weight']
    products.append(j)
    value[j] = int(v)
    weight[j] = int(w)
if set(capacity.keys()) != set(bookshelves):
    raise ValueError('Mismatch in bookshelf IDs and capacities')
if set(value.keys()) != set(products) or set(weight.keys()) != set(products):
    raise ValueError('Mismatch in product data')
m = gp.Model('Bookstore_Allocation')
x = m.addVars(bookshelves, products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value[j] * x[i, j] for i in bookshelves for j in products)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weight[j] * x[i, j] for j in products)) <= capacity[i] for i in bookshelves), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')