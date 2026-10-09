Let:
- i index the bookshelves, with BookshelfID from capacity.csv (i ∈ {1,2,3,4,5,6,7,8,9,10})
- j index the books, with ProductName from products.csv (j ∈ {The Great Gatsby, To Kill a Mockingbird, 1984, Pride and Prejudice, The Catcher in the Rye, Moby Dick, Jane Eyre, War and Peace, The Odyssey, Crime and Punishment, The Hobbit, Brave New World, Anna Karenina, Wuthering Heights, The Divine Comedy, The Iliad, Les Misérables, Dracula, Frankenstein, The Brothers Karamazov, Don Quixote, One Hundred Years of Solitude, Ulysses, The Alchemist, Meditations})

Decision variables:
x_{i,j} = number of units of book j placed on bookshelf i
x_{i,j} ∈ {0, 1, 2, ...} (nonnegative integers), for all i, j

Parameters:
Let C_i = capacity of bookshelf i (from capacity.csv)
Let v_j = value of book j (from products.csv)
Let w_j = weight of book j (from products.csv)

Data:
From capacity.csv:
BookshelfID | Capacity
1 | 200
2 | 200
3 | 300
4 | 400
5 | 550
6 | 600
7 | 650
8 | 750
9 | 820
10 | 570

From products.csv:
ProductName | Value | Weight
The Great Gatsby | 50 | 10
To Kill a Mockingbird | 70 | 20
1984 | 30 | 5
Pride and Prejudice | 60 | 15
The Catcher in the Rye | 80 | 25
Moby Dick | 90 | 30
Jane Eyre | 40 | 12
War and Peace | 100 | 35
The Odyssey | 55 | 10
Crime and Punishment | 75 | 20
The Hobbit | 65 | 18
Brave New World | 95 | 28
Anna Karenina | 45 | 8
Wuthering Heights | 85 | 22
The Divine Comedy | 70 | 25
The Iliad | 110 | 40
Les Misérables | 50 | 14
Dracula | 60 | 16
Frankenstein | 120 | 50
The Brothers Karamazov | 100 | 30
Don Quixote | 52 | 11
One Hundred Years of Solitude | 68 | 19
Ulysses | 38 | 7
The Alchemist | 58 | 14
Meditations | 82 | 24

Mathematical Model:

Maximize total value:
\[
\text{Maximize} \quad \sum_{i=1}^{10} \sum_{j=1}^{25} v_j \cdot x_{i,j}
\]
where v_j is the value of book j as listed above.

Subject to bookshelf capacity constraints:
For each bookshelf i (BookshelfID from 1 to 10, with C_i as below):
\[
\sum_{j=1}^{25} w_j \cdot x_{i,j} \leq C_i
\]
where w_j is the weight of book j as listed above, and C_i is the capacity of bookshelf i as listed above.

Variable domains:
\[
x_{i,j} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,25\}
\]

Where:
- For each bookshelf i, C_i is:
    - C_1 = 200
    - C_2 = 200
    - C_3 = 300
    - C_4 = 400
    - C_5 = 550
    - C_6 = 600
    - C_7 = 650
    - C_8 = 750
    - C_9 = 820
    - C_{10} = 570

- For each book j, (ProductName, v_j, w_j) is as listed in products.csv above.

Summary:
Maximize the total value of books placed on all bookshelves, by choosing integer numbers of each book for each bookshelf, such that the total weight of books on each bookshelf does not exceed its capacity. All data and indices are as provided in the CSVs.