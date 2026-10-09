import gurobipy as gp
from gurobipy import GRB
bookshelves = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
books = ['The Great Gatsby', 'To Kill a Mockingbird', '1984', 'Pride and Prejudice', 'The Catcher in the Rye', 'Moby Dick', 'Jane Eyre', 'War and Peace', 'The Odyssey', 'Crime and Punishment', 'The Hobbit', 'Brave New World', 'Anna Karenina', 'Wuthering Heights', 'The Divine Comedy', 'The Iliad', 'Les Misérables', 'Dracula', 'Frankenstein', 'The Brothers Karamazov', 'Don Quixote', 'One Hundred Years of Solitude', 'Ulysses', 'The Alchemist', 'Meditations']
capacities = {'1': 200, '2': 200, '3': 300, '4': 400, '5': 550, '6': 600, '7': 650, '8': 750, '9': 820, '10': 570}
values = {'The Great Gatsby': 50, 'To Kill a Mockingbird': 70, '1984': 30, 'Pride and Prejudice': 60, 'The Catcher in the Rye': 80, 'Moby Dick': 90, 'Jane Eyre': 40, 'War and Peace': 100, 'The Odyssey': 55, 'Crime and Punishment': 75, 'The Hobbit': 65, 'Brave New World': 95, 'Anna Karenina': 45, 'Wuthering Heights': 85, 'The Divine Comedy': 70, 'The Iliad': 110, 'Les Misérables': 50, 'Dracula': 60, 'Frankenstein': 120, 'The Brothers Karamazov': 100, 'Don Quixote': 52, 'One Hundred Years of Solitude': 68, 'Ulysses': 38, 'The Alchemist': 58, 'Meditations': 82}
weights = {'The Great Gatsby': 10, 'To Kill a Mockingbird': 20, '1984': 5, 'Pride and Prejudice': 15, 'The Catcher in the Rye': 25, 'Moby Dick': 30, 'Jane Eyre': 12, 'War and Peace': 35, 'The Odyssey': 10, 'Crime and Punishment': 20, 'The Hobbit': 18, 'Brave New World': 28, 'Anna Karenina': 8, 'Wuthering Heights': 22, 'The Divine Comedy': 25, 'The Iliad': 40, 'Les Misérables': 14, 'Dracula': 16, 'Frankenstein': 50, 'The Brothers Karamazov': 30, 'Don Quixote': 11, 'One Hundred Years of Solitude': 19, 'Ulysses': 7, 'The Alchemist': 14, 'Meditations': 24}
if set(capacities.keys()) != set(bookshelves):
    raise ValueError('Mismatch between bookshelf IDs and capacities keys')
if set(values.keys()) != set(books) or set(weights.keys()) != set(books):
    raise ValueError('Mismatch between book names and values/weights keys')
m = gp.Model('Bookstore_Allocation')
x_vars = m.addVars(bookshelves, books, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((values[book] * x_vars[shelf, book] for shelf in bookshelves for book in books)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weights[book] * x_vars[shelf, book] for book in books)) <= capacities[shelf] for shelf in bookshelves), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')