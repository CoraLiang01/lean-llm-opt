LEGACY_OBSERVATION = 'capacity.csv\nBookshelfID,Capacity\n1,200\n2,200\n3,300\n4,400\n5,550\n6,600\n7,650\n8,750\n9,820\n10,570\n\nproducts.csv\nProductName,Value,Weight\nThe Great Gatsby,50,10\nTo Kill a Mockingbird,70,20\n1984,30,5\nPride and Prejudice,60,15\nThe Catcher in the Rye,80,25\nMoby Dick,90,30\nJane Eyre,40,12\nWar and Peace,100,35\nThe Odyssey,55,10\nCrime and Punishment,75,20\nThe Hobbit,65,18\nBrave New World,95,28\nAnna Karenina,45,8\nWuthering Heights,85,22\nThe Divine Comedy,70,25\nThe Iliad,110,40\nLes Misérables,50,14\nDracula,60,16\nFrankenstein,120,50\nThe Brothers Karamazov,100,30\nDon Quixote,52,11\nOne Hundred Years of Solitude,68,19\nUlysses,38,7\nThe Alchemist,58,14\nMeditations,82,24'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'BookshelfID': '1', 'Capacity': '200'}}, {'source': 'capacity.csv', 'values': {'BookshelfID': '2', 'Capacity': '200'}}, {'source': 'capacity.csv', 'values': {'BookshelfID': '3', 'Capacity': '300'}}, {'source': 'capacity.csv', 'values': {'BookshelfID': '4', 'Capacity': '400'}}, {'source': 'capacity.csv', 'values': {'BookshelfID': '5', 'Capacity': '550'}}, {'source': 'capacity.csv', 'values': {'BookshelfID': '6', 'Capacity': '600'}}, {'source': 'capacity.csv', 'values': {'BookshelfID': '7', 'Capacity': '650'}}, {'source': 'capacity.csv', 'values': {'BookshelfID': '8', 'Capacity': '750'}}, {'source': 'capacity.csv', 'values': {'BookshelfID': '9', 'Capacity': '820'}}, {'source': 'capacity.csv', 'values': {'BookshelfID': '10', 'Capacity': '570'}}, {'source': 'products.csv', 'values': {'ProductName': 'The Great Gatsby', 'Value': '50', 'Weight': '10'}}, {'source': 'products.csv', 'values': {'ProductName': 'To Kill a Mockingbird', 'Value': '70', 'Weight': '20'}}, {'source': 'products.csv', 'values': {'ProductName': '1984', 'Value': '30', 'Weight': '5'}}, {'source': 'products.csv', 'values': {'ProductName': 'Pride and Prejudice', 'Value': '60', 'Weight': '15'}}, {'source': 'products.csv', 'values': {'ProductName': 'The Catcher in the Rye', 'Value': '80', 'Weight': '25'}}, {'source': 'products.csv', 'values': {'ProductName': 'Moby Dick', 'Value': '90', 'Weight': '30'}}, {'source': 'products.csv', 'values': {'ProductName': 'Jane Eyre', 'Value': '40', 'Weight': '12'}}, {'source': 'products.csv', 'values': {'ProductName': 'War and Peace', 'Value': '100', 'Weight': '35'}}, {'source': 'products.csv', 'values': {'ProductName': 'The Odyssey', 'Value': '55', 'Weight': '10'}}, {'source': 'products.csv', 'values': {'ProductName': 'Crime and Punishment', 'Value': '75', 'Weight': '20'}}, {'source': 'products.csv', 'values': {'ProductName': 'The Hobbit', 'Value': '65', 'Weight': '18'}}, {'source': 'products.csv', 'values': {'ProductName': 'Brave New World', 'Value': '95', 'Weight': '28'}}, {'source': 'products.csv', 'values': {'ProductName': 'Anna Karenina', 'Value': '45', 'Weight': '8'}}, {'source': 'products.csv', 'values': {'ProductName': 'Wuthering Heights', 'Value': '85', 'Weight': '22'}}, {'source': 'products.csv', 'values': {'ProductName': 'The Divine Comedy', 'Value': '70', 'Weight': '25'}}, {'source': 'products.csv', 'values': {'ProductName': 'The Iliad', 'Value': '110', 'Weight': '40'}}, {'source': 'products.csv', 'values': {'ProductName': 'Les Misérables', 'Value': '50', 'Weight': '14'}}, {'source': 'products.csv', 'values': {'ProductName': 'Dracula', 'Value': '60', 'Weight': '16'}}, {'source': 'products.csv', 'values': {'ProductName': 'Frankenstein', 'Value': '120', 'Weight': '50'}}, {'source': 'products.csv', 'values': {'ProductName': 'The Brothers Karamazov', 'Value': '100', 'Weight': '30'}}, {'source': 'products.csv', 'values': {'ProductName': 'Don Quixote', 'Value': '52', 'Weight': '11'}}, {'source': 'products.csv', 'values': {'ProductName': 'One Hundred Years of Solitude', 'Value': '68', 'Weight': '19'}}, {'source': 'products.csv', 'values': {'ProductName': 'Ulysses', 'Value': '38', 'Weight': '7'}}, {'source': 'products.csv', 'values': {'ProductName': 'The Alchemist', 'Value': '58', 'Weight': '14'}}, {'source': 'products.csv', 'values': {'ProductName': 'Meditations', 'Value': '82', 'Weight': '24'}}]
from gurobipy import Model, GRB
bookshelf_records = [r for r in LEGACY_RECORDS if r['source'] == 'capacity.csv']
product_records = [r for r in LEGACY_RECORDS if r['source'] == 'products.csv']
bookshelves = {}
for rec in bookshelf_records:
    bid = rec['values']['BookshelfID']
    cap = int(rec['values']['Capacity'])
    if bid in bookshelves:
        raise ValueError(f'Duplicate BookshelfID: {bid}')
    bookshelves[bid] = cap
products = {}
for rec in product_records:
    pname = rec['values']['ProductName']
    val = int(rec['values']['Value'])
    wt = int(rec['values']['Weight'])
    if pname in products:
        raise ValueError(f'Duplicate ProductName: {pname}')
    products[pname] = {'Value': val, 'Weight': wt}
if len(bookshelves) != 10:
    raise ValueError('Expected 10 bookshelves, got %d' % len(bookshelves))
if len(products) != 25:
    raise ValueError('Expected 25 products, got %d' % len(products))
I = list(bookshelves.keys())
J = list(products.keys())
m = Model()
m.setParam('MIPGap', 0.0001)
x = m.addVars(I, J, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(sum((products[j]['Value'] * x[i, j] for i in I for j in J)), GRB.MAXIMIZE)
for i in I:
    m.addConstr(sum((products[j]['Weight'] * x[i, j] for j in J)) <= bookshelves[i], name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print('ObjVal:', m.ObjVal)
    for i in I:
        for j in J:
            v = x[i, j].X
            print(f'{x[i, j].VarName} {v}')
else:
    print('Solver status:', m.Status)