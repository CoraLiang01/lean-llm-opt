Let $x_{ij}$ be the number of units of book $j$ (with ProductName as below) to be placed on bookshelf $i$ (with BookshelfID as below). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- Bookshelves (indexed by $i$, BookshelfID):  
  1, 2, 3, 4, 5, 6, 7, 8, 9, 10

- Books (indexed by $j$, ProductName):  
  The Great Gatsby, To Kill a Mockingbird, 1984, Pride and Prejudice, The Catcher in the Rye, Moby Dick, Jane Eyre, War and Peace, The Odyssey, Crime and Punishment, The Hobbit, Brave New World, Anna Karenina, Wuthering Heights, The Divine Comedy, The Iliad, Les Misérables, Dracula, Frankenstein, The Brothers Karamazov, Don Quixote, One Hundred Years of Solitude, Ulysses, The Alchemist, Meditations

- Book values $v_j$ and weights $w_j$:

| ProductName                      | Value | Weight |
|----------------------------------|-------|--------|
| The Great Gatsby                 | 50    | 10     |
| To Kill a Mockingbird            | 70    | 20     |
| 1984                             | 30    | 5      |
| Pride and Prejudice              | 60    | 15     |
| The Catcher in the Rye           | 80    | 25     |
| Moby Dick                        | 90    | 30     |
| Jane Eyre                        | 40    | 12     |
| War and Peace                    | 100   | 35     |
| The Odyssey                      | 55    | 10     |
| Crime and Punishment             | 75    | 20     |
| The Hobbit                       | 65    | 18     |
| Brave New World                  | 95    | 28     |
| Anna Karenina                    | 45    | 8      |
| Wuthering Heights                | 85    | 22     |
| The Divine Comedy                | 70    | 25     |
| The Iliad                        | 110   | 40     |
| Les Misérables                   | 50    | 14     |
| Dracula                          | 60    | 16     |
| Frankenstein                     | 120   | 50     |
| The Brothers Karamazov           | 100   | 30     |
| Don Quixote                      | 52    | 11     |
| One Hundred Years of Solitude    | 68    | 19     |
| Ulysses                          | 38    | 7      |
| The Alchemist                    | 58    | 14     |
| Meditations                      | 82    | 24     |

- Bookshelf capacities $C_i$:

| BookshelfID | Capacity |
|-------------|----------|
| 1           | 200      |
| 2           | 200      |
| 3           | 300      |
| 4           | 400      |
| 5           | 550      |
| 6           | 600      |
| 7           | 650      |
| 8           | 750      |
| 9           | 820      |
| 10          | 570      |

---

**Mathematical Model:**

**Objective:**
\[
\max \sum_{i \in \{1,\ldots,10\}} \sum_{j \in \text{Books}} v_j \cdot x_{ij}
\]

**Subject to:**

For each bookshelf $i$ (BookshelfID as above):
\[
\sum_{j \in \text{Books}} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,10\}
\]

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,10\},\ \forall j \in \text{Books}
\]

---

**Where:**

- $x_{ij}$: Number of units of book $j$ to place on bookshelf $i$ (integer, $\geq 0$)
- $v_j$: Value of book $j$ (see table above)
- $w_j$: Weight of book $j$ (see table above)
- $C_i$: Capacity of bookshelf $i$ (see table above)
- BookshelfID and ProductName are as listed above, and all indices and coefficients are as retrieved.

**All data and identifiers are used exactly as retrieved and in source order.**