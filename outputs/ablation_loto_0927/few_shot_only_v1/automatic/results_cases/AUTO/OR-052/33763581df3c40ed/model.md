**Mathematical Optimization Model**

**Sets and Indices:**
- Let $I$ be the set of bookshelves, indexed by $i$, with BookshelfID from the "capacity.csv" file.
- Let $J$ be the set of books, indexed by $j$, with ProductName from the "products.csv" file.

**Parameters:**
- $C_i$: Capacity of bookshelf $i$ (from "capacity.csv").
- $v_j$: Value of book $j$ (from "products.csv").
- $w_j$: Weight of book $j$ (from "products.csv").

**Decision Variables:**
- $x_{ij}$: Number of units of book $j$ to place on bookshelf $i$.
- $x_{ij} \in \mathbb{Z}_{\geq 0}$ (nonnegative integers), for all $i \in I$, $j \in J$.

---

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
\]

---

**Subject to:**

**1. Bookshelf Capacity Constraints:**
\[
\sum_{j \in J} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in I
\]

**2. Nonnegativity and Integrality:**
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
\]

---

**Parameter Data (from CSVs):**

- **Bookshelves and Capacities:**

| BookshelfID | $C_i$ (Capacity) |
|-------------|------------------|
| 1           | 200              |
| 2           | 200              |
| 3           | 300              |
| 4           | 400              |
| 5           | 550              |
| 6           | 600              |
| 7           | 650              |
| 8           | 750              |
| 9           | 820              |
| 10          | 570              |

- **Books, Values, and Weights:**

| ProductName                        | $v_j$ (Value) | $w_j$ (Weight) |
|------------------------------------|---------------|---------------|
| The Great Gatsby                   | 50            | 10            |
| To Kill a Mockingbird              | 70            | 20            |
| 1984                               | 30            | 5             |
| Pride and Prejudice                | 60            | 15            |
| The Catcher in the Rye             | 80            | 25            |
| Moby Dick                          | 90            | 30            |
| Jane Eyre                          | 40            | 12            |
| War and Peace                      | 100           | 35            |
| The Odyssey                        | 55            | 10            |
| Crime and Punishment               | 75            | 20            |
| The Hobbit                         | 65            | 18            |
| Brave New World                    | 95            | 28            |
| Anna Karenina                      | 45            | 8             |
| Wuthering Heights                  | 85            | 22            |
| The Divine Comedy                  | 70            | 25            |
| The Iliad                          | 110           | 40            |
| Les Misérables                     | 50            | 14            |
| Dracula                            | 60            | 16            |
| Frankenstein                       | 120           | 50            |
| The Brothers Karamazov             | 100           | 30            |
| Don Quixote                        | 52            | 11            |
| One Hundred Years of Solitude      | 68            | 19            |
| Ulysses                            | 38            | 7             |
| The Alchemist                      | 58            | 14            |
| Meditations                        | 82            | 24            |

---

**Full Model:**

\[
\begin{align*}
\max\quad & \sum_{i=1}^{10} \sum_{j=1}^{25} v_j \cdot x_{ij} \\
\text{s.t.}\quad & \sum_{j=1}^{25} w_j \cdot x_{ij} \leq C_i \qquad \forall i = 1,\ldots,10 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,10,\; j = 1,\ldots,25
\end{align*}
\]

Where $v_j$ and $w_j$ are as listed above, and $C_i$ is the capacity for each bookshelf.