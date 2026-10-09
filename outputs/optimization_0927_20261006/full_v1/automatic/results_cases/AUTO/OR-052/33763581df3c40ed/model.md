Let $x_{ij}$ be the number of units of book $j$ to be placed on bookshelf $i$. All $x_{ij}$ are integer and $\geq 0$.

Let $B$ be the set of bookshelves (indexed by BookshelfID), and $P$ the set of books (indexed by ProductName).

Let $c_i$ be the capacity of bookshelf $i$ (from capacity.csv).

Let $v_j$ be the value and $w_j$ the weight of book $j$ (from products.csv).

---

**Objective:**

$$
\max \sum_{i \in B} \sum_{j \in P} v_j \cdot x_{ij}
$$

**Subject to:**

For each bookshelf $i \in B$:
$$
\sum_{j \in P} w_j \cdot x_{ij} \leq c_i
$$

For all $i \in B$, $j \in P$:
$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

---

**Parameter Data (in source order):**

Bookshelves (from capacity.csv):

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

Books (from products.csv):

| ProductName                      | Value | Weight |
|-----------------------------------|-------|--------|
| The Great Gatsby                  | 50    | 10     |
| To Kill a Mockingbird             | 70    | 20     |
| 1984                              | 30    | 5      |
| Pride and Prejudice               | 60    | 15     |
| The Catcher in the Rye            | 80    | 25     |
| Moby Dick                         | 90    | 30     |
| Jane Eyre                         | 40    | 12     |
| War and Peace                     | 100   | 35     |
| The Odyssey                       | 55    | 10     |
| Crime and Punishment              | 75    | 20     |
| The Hobbit                        | 65    | 18     |
| Brave New World                   | 95    | 28     |
| Anna Karenina                     | 45    | 8      |
| Wuthering Heights                 | 85    | 22     |
| The Divine Comedy                 | 70    | 25     |
| The Iliad                         | 110   | 40     |
| Les Misérables                    | 50    | 14     |
| Dracula                           | 60    | 16     |
| Frankenstein                      | 120   | 50     |
| The Brothers Karamazov            | 100   | 30     |
| Don Quixote                       | 52    | 11     |
| One Hundred Years of Solitude     | 68    | 19     |
| Ulysses                           | 38    | 7      |
| The Alchemist                     | 58    | 14     |
| Meditations                       | 82    | 24     |

---

**Decision variables:**

$x_{ij}$: integer, $\geq 0$, for all $i \in$ BookshelfID, $j \in$ ProductName.

---

**Complete Model:**

$$
\begin{align*}
\max \quad & \sum_{i \in \{1,\ldots,10\}} \sum_{j \in P} v_j x_{ij} \\
\text{s.t.} \quad & \sum_{j \in P} w_j x_{ij} \leq c_i, \quad \forall i \in \{1,\ldots,10\} \\
& x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in \{1,\ldots,10\},\ j \in P
\end{align*}
$$

Where $P$ and all coefficients are as listed above.