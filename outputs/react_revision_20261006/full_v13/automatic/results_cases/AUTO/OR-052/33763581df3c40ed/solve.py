CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'In the context of a bookstore, the store needs to allocate various types of books into different '
          'bookshelves. Specifically, the store has several bookshelves, each with a capacity limit provided in '
          '“capacity.csv.” The predefined value and weight of each book can be found in “products.csv.” The objective '
          'is to determine the optimal number of units of each book to place on each bookshelf to maximize the total '
          'value of the books across all bookshelves while ensuring that the total weight of the books on each '
          'bookshelf does not exceed its capacity. The decision variables x_ij represent the number of units of book j '
          'to be placed on bookshelf i.The decision variables must be integers.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'columns': ['BookshelfID', 'Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0, 'values': {'BookshelfID': '1', 'Capacity': '200'}},
                         {'source_row': 1, 'values': {'BookshelfID': '2', 'Capacity': '200'}},
                         {'source_row': 2, 'values': {'BookshelfID': '3', 'Capacity': '300'}},
                         {'source_row': 3, 'values': {'BookshelfID': '4', 'Capacity': '400'}},
                         {'source_row': 4, 'values': {'BookshelfID': '5', 'Capacity': '550'}},
                         {'source_row': 5, 'values': {'BookshelfID': '6', 'Capacity': '600'}},
                         {'source_row': 6, 'values': {'BookshelfID': '7', 'Capacity': '650'}},
                         {'source_row': 7, 'values': {'BookshelfID': '8', 'Capacity': '750'}},
                         {'source_row': 8, 'values': {'BookshelfID': '9', 'Capacity': '820'}},
                         {'source_row': 9, 'values': {'BookshelfID': '10', 'Capacity': '570'}}],
             'returned_rows': 10,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['ProductName', 'Value', 'Weight'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 25,
             'records': [{'source_row': 0,
                          'values': {'ProductName': 'The Great Gatsby', 'Value': '50', 'Weight': '10'}},
                         {'source_row': 1,
                          'values': {'ProductName': 'To Kill a Mockingbird', 'Value': '70', 'Weight': '20'}},
                         {'source_row': 2, 'values': {'ProductName': '1984', 'Value': '30', 'Weight': '5'}},
                         {'source_row': 3,
                          'values': {'ProductName': 'Pride and Prejudice', 'Value': '60', 'Weight': '15'}},
                         {'source_row': 4,
                          'values': {'ProductName': 'The Catcher in the Rye', 'Value': '80', 'Weight': '25'}},
                         {'source_row': 5, 'values': {'ProductName': 'Moby Dick', 'Value': '90', 'Weight': '30'}},
                         {'source_row': 6, 'values': {'ProductName': 'Jane Eyre', 'Value': '40', 'Weight': '12'}},
                         {'source_row': 7, 'values': {'ProductName': 'War and Peace', 'Value': '100', 'Weight': '35'}},
                         {'source_row': 8, 'values': {'ProductName': 'The Odyssey', 'Value': '55', 'Weight': '10'}},
                         {'source_row': 9,
                          'values': {'ProductName': 'Crime and Punishment', 'Value': '75', 'Weight': '20'}},
                         {'source_row': 10, 'values': {'ProductName': 'The Hobbit', 'Value': '65', 'Weight': '18'}},
                         {'source_row': 11,
                          'values': {'ProductName': 'Brave New World', 'Value': '95', 'Weight': '28'}},
                         {'source_row': 12, 'values': {'ProductName': 'Anna Karenina', 'Value': '45', 'Weight': '8'}},
                         {'source_row': 13,
                          'values': {'ProductName': 'Wuthering Heights', 'Value': '85', 'Weight': '22'}},
                         {'source_row': 14,
                          'values': {'ProductName': 'The Divine Comedy', 'Value': '70', 'Weight': '25'}},
                         {'source_row': 15, 'values': {'ProductName': 'The Iliad', 'Value': '110', 'Weight': '40'}},
                         {'source_row': 16, 'values': {'ProductName': 'Les Misérables', 'Value': '50', 'Weight': '14'}},
                         {'source_row': 17, 'values': {'ProductName': 'Dracula', 'Value': '60', 'Weight': '16'}},
                         {'source_row': 18, 'values': {'ProductName': 'Frankenstein', 'Value': '120', 'Weight': '50'}},
                         {'source_row': 19,
                          'values': {'ProductName': 'The Brothers Karamazov', 'Value': '100', 'Weight': '30'}},
                         {'source_row': 20, 'values': {'ProductName': 'Don Quixote', 'Value': '52', 'Weight': '11'}},
                         {'source_row': 21,
                          'values': {'ProductName': 'One Hundred Years of Solitude', 'Value': '68', 'Weight': '19'}},
                         {'source_row': 22, 'values': {'ProductName': 'Ulysses', 'Value': '38', 'Weight': '7'}},
                         {'source_row': 23, 'values': {'ProductName': 'The Alchemist', 'Value': '58', 'Weight': '14'}},
                         {'source_row': 24, 'values': {'ProductName': 'Meditations', 'Value': '82', 'Weight': '24'}}],
             'returned_rows': 25,
             'role': 'file_1',
             'table_id': 'file_1_view_0'}],
 'validation': {'fallback_reason': "Relationship references an unknown table_id: {'type': 'matrix', 'matrix_table_id': "
                                   "'file_2_view_0', 'row_id_column': 'BookshelfID', 'row_axis': {'table_id': "
                                   "'file_0_view_0', 'id_column': 'BookshelfID'}, 'column_axis': {'table_id': "
                                   "'file_1_view_0', 'id_column': 'ProductName'}}",
                'planner_errors': ["Relationship references an unknown table_id: {'type': 'matrix', 'matrix_table_id': "
                                   "'file_2_view_0', 'row_id_column': 'BookshelfID', 'row_axis': {'table_id': "
                                   "'file_0_view_0', 'id_column': 'BookshelfID'}, 'column_axis': {'table_id': "
                                   "'file_1_view_0', 'id_column': 'ProductName'}}"],
                'status': 'FALLBACK_FULL_DATA'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem(CSVQA_FRAMES):
    bookshelf_frame = CSVQA_FRAMES['file_0_view_0']
    I = []
    c = {}
    for (_, row) in bookshelf_frame.iterrows():
        bookshelf_id = row['BookshelfID']
        I.append(bookshelf_id)
        try:
            c[bookshelf_id] = float(row['Capacity'])
        except Exception:
            raise ValueError(f"Invalid Capacity for BookshelfID {bookshelf_id}: {row['Capacity']}")
    product_frame = CSVQA_FRAMES['file_1_view_0']
    J = []
    v = {}
    w = {}
    for (_, row) in product_frame.iterrows():
        product_name = row['ProductName']
        J.append(product_name)
        try:
            v[product_name] = float(row['Value'])
        except Exception:
            raise ValueError(f"Invalid Value for ProductName {product_name}: {row['Value']}")
        try:
            w[product_name] = float(row['Weight'])
        except Exception:
            raise ValueError(f"Invalid Weight for ProductName {product_name}: {row['Weight']}")
    if len(I) == 0 or len(J) == 0:
        raise ValueError('No bookshelves or products found in the data.')
    m = gp.Model('bookshelf_allocation')
    x_keys = [(i, j) for i in I for j in J]
    quantity_vars = m.addVars(x_keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v[j] * quantity_vars[i, j] for i in I for j in J)), GRB.MAXIMIZE)
    m.addConstrs((gp.quicksum((w[j] * quantity_vars[i, j] for j in J)) <= c[i] for i in I), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')