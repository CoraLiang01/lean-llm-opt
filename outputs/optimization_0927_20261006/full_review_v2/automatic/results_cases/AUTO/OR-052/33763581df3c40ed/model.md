Let $x_{ij}$ be the number of units of book $j$ to be placed on bookshelf $i$. All $x_{ij}$ are required to be nonnegative integers.

Let:
- $i$ index the bookshelves, with BookshelfID as below.
- $j$ index the books, with ProductName as below.
- $v_j$ = Value of book $j$
- $w_j$ = Weight of book $j$
- $C_i$ = Capacity of bookshelf $i$

#### Sets and Parameters

Bookshelves (from capacity.csv, in source order):

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

Books (from products.csv, in source order):

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

#### Decision Variables

$x_{ij} \in \mathbb{Z}_{\geq 0}$, for all BookshelfID $i$ and ProductName $j$

#### Objective Function

$$
\max \sum_{i \in \{\text{BookshelfID}\}} \sum_{j \in \{\text{ProductName}\}} v_j \cdot x_{ij}
$$

#### Constraints

For each bookshelf $i$ (BookshelfID):

$$
\sum_{j \in \{\text{ProductName}\}} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{\text{BookshelfID}\}
$$

For all $i, j$:

$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

#### Explicit Data

Bookshelf capacities (in source order):

- $C_1 = 200$
- $C_2 = 200$
- $C_3 = 300$
- $C_4 = 400$
- $C_5 = 550$
- $C_6 = 600$
- $C_7 = 650$
- $C_8 = 750$
- $C_9 = 820$
- $C_{10} = 570$

Book values and weights (in source order):

- The Great Gatsby: $v = 50$, $w = 10$
- To Kill a Mockingbird: $v = 70$, $w = 20$
- 1984: $v = 30$, $w = 5$
- Pride and Prejudice: $v = 60$, $w = 15$
- The Catcher in the Rye: $v = 80$, $w = 25$
- Moby Dick: $v = 90$, $w = 30$
- Jane Eyre: $v = 40$, $w = 12$
- War and Peace: $v = 100$, $w = 35$
- The Odyssey: $v = 55$, $w = 10$
- Crime and Punishment: $v = 75$, $w = 20$
- The Hobbit: $v = 65$, $w = 18$
- Brave New World: $v = 95$, $w = 28$
- Anna Karenina: $v = 45$, $w = 8$
- Wuthering Heights: $v = 85$, $w = 22$
- The Divine Comedy: $v = 70$, $w = 25$
- The Iliad: $v = 110$, $w = 40$
- Les Misérables: $v = 50$, $w = 14$
- Dracula: $v = 60$, $w = 16$
- Frankenstein: $v = 120$, $w = 50$
- The Brothers Karamazov: $v = 100$, $w = 30$
- Don Quixote: $v = 52$, $w = 11$
- One Hundred Years of Solitude: $v = 68$, $w = 19$
- Ulysses: $v = 38$, $w = 7$
- The Alchemist: $v = 58$, $w = 14$
- Meditations: $v = 82$, $w = 24$

#### Complete Model

$$
\begin{align*}
\max \quad & \sum_{i=1}^{10} \sum_{j=1}^{25} v_j \cdot x_{ij} \\
\text{s.t.} \quad & \sum_{j=1}^{25} w_j \cdot x_{ij} \leq C_i, \quad \forall i = 1,\ldots,10 \\
& x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i = 1,\ldots,10,\ j = 1,\ldots,25
\end{align*}
$$

where $v_j$ and $w_j$ are as listed above, and $C_i$ as above, with $i$ corresponding to BookshelfID and $j$ to ProductName in the given order.