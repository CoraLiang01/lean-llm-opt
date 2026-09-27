LEGACY_OBSERVATION = '{"values": {"BookshelfID": "1", "Capacity": "200"}}\n{"values": {"BookshelfID": "2", "Capacity": "200"}}\n{"values": {"BookshelfID": "3", "Capacity": "300"}}\n{"values": {"BookshelfID": "4", "Capacity": "400"}}\n{"values": {"BookshelfID": "5", "Capacity": "550"}}\n{"values": {"BookshelfID": "6", "Capacity": "600"}}\n{"values": {"BookshelfID": "7", "Capacity": "650"}}\n{"values": {"BookshelfID": "8", "Capacity": "750"}}\n{"values": {"BookshelfID": "9", "Capacity": "820"}}\n{"values": {"BookshelfID": "10", "Capacity": "570"}}\n{"values": {"ProductName": "The Great Gatsby", "Value": "50", "Weight": "10"}}\n{"values": {"ProductName": "To Kill a Mockingbird", "Value": "70", "Weight": "20"}}\n{"values": {"ProductName": "1984", "Value": "30", "Weight": "5"}}\n{"values": {"ProductName": "Pride and Prejudice", "Value": "60", "Weight": "15"}}\n{"values": {"ProductName": "The Catcher in the Rye", "Value": "80", "Weight": "25"}}\n{"values": {"ProductName": "Moby Dick", "Value": "90", "Weight": "30"}}\n{"values": {"ProductName": "Jane Eyre", "Value": "40", "Weight": "12"}}\n{"values": {"ProductName": "War and Peace", "Value": "100", "Weight": "35"}}\n{"values": {"ProductName": "The Odyssey", "Value": "55", "Weight": "10"}}\n{"values": {"ProductName": "Crime and Punishment", "Value": "75", "Weight": "20"}}\n{"values": {"ProductName": "The Hobbit", "Value": "65", "Weight": "18"}}\n{"values": {"ProductName": "Brave New World", "Value": "95", "Weight": "28"}}\n{"values": {"ProductName": "Anna Karenina", "Value": "45", "Weight": "8"}}\n{"values": {"ProductName": "Wuthering Heights", "Value": "85", "Weight": "22"}}\n{"values": {"ProductName": "The Divine Comedy", "Value": "70", "Weight": "25"}}\n{"values": {"ProductName": "The Iliad", "Value": "110", "Weight": "40"}}\n{"values": {"ProductName": "Les Misérables", "Value": "50", "Weight": "14"}}\n{"values": {"ProductName": "Dracula", "Value": "60", "Weight": "16"}}\n{"values": {"ProductName": "Frankenstein", "Value": "120", "Weight": "50"}}\n{"values": {"ProductName": "The Brothers Karamazov", "Value": "100", "Weight": "30"}}\n{"values": {"ProductName": "Don Quixote", "Value": "52", "Weight": "11"}}\n{"values": {"ProductName": "One Hundred Years of Solitude", "Value": "68", "Weight": "19"}}\n{"values": {"ProductName": "Ulysses", "Value": "38", "Weight": "7"}}\n{"values": {"ProductName": "The Alchemist", "Value": "58", "Weight": "14"}}\n{"values": {"ProductName": "Meditations", "Value": "82", "Weight": "24"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'BookshelfID': '1', 'Capacity': '200'}}, {'source': '', 'values': {'BookshelfID': '2', 'Capacity': '200'}}, {'source': '', 'values': {'BookshelfID': '3', 'Capacity': '300'}}, {'source': '', 'values': {'BookshelfID': '4', 'Capacity': '400'}}, {'source': '', 'values': {'BookshelfID': '5', 'Capacity': '550'}}, {'source': '', 'values': {'BookshelfID': '6', 'Capacity': '600'}}, {'source': '', 'values': {'BookshelfID': '7', 'Capacity': '650'}}, {'source': '', 'values': {'BookshelfID': '8', 'Capacity': '750'}}, {'source': '', 'values': {'BookshelfID': '9', 'Capacity': '820'}}, {'source': '', 'values': {'BookshelfID': '10', 'Capacity': '570'}}, {'source': '', 'values': {'ProductName': 'The Great Gatsby', 'Value': '50', 'Weight': '10'}}, {'source': '', 'values': {'ProductName': 'To Kill a Mockingbird', 'Value': '70', 'Weight': '20'}}, {'source': '', 'values': {'ProductName': '1984', 'Value': '30', 'Weight': '5'}}, {'source': '', 'values': {'ProductName': 'Pride and Prejudice', 'Value': '60', 'Weight': '15'}}, {'source': '', 'values': {'ProductName': 'The Catcher in the Rye', 'Value': '80', 'Weight': '25'}}, {'source': '', 'values': {'ProductName': 'Moby Dick', 'Value': '90', 'Weight': '30'}}, {'source': '', 'values': {'ProductName': 'Jane Eyre', 'Value': '40', 'Weight': '12'}}, {'source': '', 'values': {'ProductName': 'War and Peace', 'Value': '100', 'Weight': '35'}}, {'source': '', 'values': {'ProductName': 'The Odyssey', 'Value': '55', 'Weight': '10'}}, {'source': '', 'values': {'ProductName': 'Crime and Punishment', 'Value': '75', 'Weight': '20'}}, {'source': '', 'values': {'ProductName': 'The Hobbit', 'Value': '65', 'Weight': '18'}}, {'source': '', 'values': {'ProductName': 'Brave New World', 'Value': '95', 'Weight': '28'}}, {'source': '', 'values': {'ProductName': 'Anna Karenina', 'Value': '45', 'Weight': '8'}}, {'source': '', 'values': {'ProductName': 'Wuthering Heights', 'Value': '85', 'Weight': '22'}}, {'source': '', 'values': {'ProductName': 'The Divine Comedy', 'Value': '70', 'Weight': '25'}}, {'source': '', 'values': {'ProductName': 'The Iliad', 'Value': '110', 'Weight': '40'}}, {'source': '', 'values': {'ProductName': 'Les Misérables', 'Value': '50', 'Weight': '14'}}, {'source': '', 'values': {'ProductName': 'Dracula', 'Value': '60', 'Weight': '16'}}, {'source': '', 'values': {'ProductName': 'Frankenstein', 'Value': '120', 'Weight': '50'}}, {'source': '', 'values': {'ProductName': 'The Brothers Karamazov', 'Value': '100', 'Weight': '30'}}, {'source': '', 'values': {'ProductName': 'Don Quixote', 'Value': '52', 'Weight': '11'}}, {'source': '', 'values': {'ProductName': 'One Hundred Years of Solitude', 'Value': '68', 'Weight': '19'}}, {'source': '', 'values': {'ProductName': 'Ulysses', 'Value': '38', 'Weight': '7'}}, {'source': '', 'values': {'ProductName': 'The Alchemist', 'Value': '58', 'Weight': '14'}}, {'source': '', 'values': {'ProductName': 'Meditations', 'Value': '82', 'Weight': '24'}}]
import gurobipy as gp
from gurobipy import GRB
bookshelf_caps = {}
products = []
value = {}
weight = {}
for rec in LEGACY_RECORDS:
    v = rec['values']
    if 'BookshelfID' in v and 'Capacity' in v:
        bookshelf_caps[v['BookshelfID']] = int(v['Capacity'])
    elif 'ProductName' in v and 'Value' in v and ('Weight' in v):
        pname = v['ProductName']
        products.append(pname)
        value[pname] = int(v['Value'])
        weight[pname] = int(v['Weight'])
bookshelves = list(bookshelf_caps.keys())
if len(bookshelves) == 0 or len(products) == 0:
    raise ValueError('Missing bookshelf or product data.')
for pname in products:
    if pname not in value or pname not in weight:
        raise ValueError(f'Missing value/weight for product {pname}.')
for bid in bookshelves:
    if bid not in bookshelf_caps:
        raise ValueError(f'Missing capacity for bookshelf {bid}.')
m = gp.Model('Bookstore_Shelf_Allocation')
x = m.addVars(bookshelves, products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value[p] * x[b, p] for b in bookshelves for p in products)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weight[p] * x[b, p] for p in products)) <= bookshelf_caps[b] for b in bookshelves), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')