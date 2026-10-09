**Sets and Indices:**
- Let $i$ index bookshelves, with BookshelfID from capacity.csv: $i \in \{1,2,3,4,5,6,7,8,9,10\}$
- Let $j$ index books, with ProductName from products.csv: $j \in \{$The Great Gatsby, To Kill a Mockingbird, 1984, Pride and Prejudice, The Catcher in the Rye, Moby Dick, Jane Eyre, War and Peace, The Odyssey, Crime and Punishment, The Hobbit, Brave New World, Anna Karenina, Wuthering Heights, The Divine Comedy, The Iliad, Les Misérables, Dracula, Frankenstein, The Brothers Karamazov, Don Quixote, One Hundred Years of Solitude, Ulysses, The Alchemist, Meditations$\}$

**Parameters:**
- $v_j$ = Value of book $j$ (from products.csv)
- $w_j$ = Weight of book $j$ (from products.csv)
- $C_i$ = Capacity of bookshelf $i$ (from capacity.csv)

**Decision Variables:**
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of units of book $j$ placed on bookshelf $i$

**Objective:**
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{25} v_j \cdot x_{ij}
\]

**Subject to:**

For each bookshelf $i$ (BookshelfID $i$):
\[
\sum_{j=1}^{25} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,2,3,4,5,6,7,8,9,10\}
\]

For all $i, j$:
\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]

---

**Numerical Data:**

*Bookshelves (from capacity.csv):*

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

*Books (from products.csv):*

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

**Full Model:**

\[
\begin{align*}
\max \quad & \sum_{i=1}^{10} \sum_{j=1}^{25} v_j \cdot x_{ij} \\
\text{s.t.} \quad & \sum_{j=1}^{25} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,2,3,4,5,6,7,8,9,10\} \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
\end{align*}
\]

Where:
- $v_j$ and $w_j$ are as listed above for each ProductName,
- $C_i$ is as listed above for each BookshelfID.