Let $x_{ij}$ be the number of units of book $j$ placed on bookshelf $i$, where $i$ indexes BookshelfID from the "capacity.csv" file and $j$ indexes ProductName from the "products.csv" file. All $x_{ij}$ are integer and $\geq 0$.

**Parameters:**

- Let $S$ be the set of bookshelves (BookshelfID):  
  $S = \{1, 2, 3, 4, 5, 6, 7, 8, 9, 10\}$
- Let $B$ be the set of books (ProductName), in the order given in "products.csv":  
  $B = \{$The Great Gatsby, To Kill a Mockingbird, 1984, Pride and Prejudice, The Catcher in the Rye, Moby Dick, Jane Eyre, War and Peace, The Odyssey, Crime and Punishment, The Hobbit, Brave New World, Anna Karenina, Wuthering Heights, The Divine Comedy, The Iliad, Les Misérables, Dracula, Frankenstein, The Brothers Karamazov, Don Quixote, One Hundred Years of Solitude, Ulysses, The Alchemist, Meditations$\}$
- For each bookshelf $i \in S$, let $C_i$ be its capacity (from "capacity.csv").
- For each book $j \in B$, let $v_j$ be its value and $w_j$ its weight (from "products.csv").

**Objective:**

Maximize the total value of books allocated:
$$
\max \sum_{i \in S} \sum_{j \in B} v_j \cdot x_{ij}
$$

**Constraints:**

1. **Shelf Capacity Constraints:**  
   For each bookshelf $i \in S$,
   $$
   \sum_{j \in B} w_j \cdot x_{ij} \leq C_i
   $$
2. **Non-negativity and Integrality:**  
   For all $i \in S$, $j \in B$,
   $$
   x_{ij} \in \mathbb{Z}_{\geq 0}
   $$

**Parameter Values (from CSVs):**

- Bookshelves and capacities:
  - 1: $C_1 = 200$
  - 2: $C_2 = 200$
  - 3: $C_3 = 300$
  - 4: $C_4 = 400$
  - 5: $C_5 = 550$
  - 6: $C_6 = 600$
  - 7: $C_7 = 650$
  - 8: $C_8 = 750$
  - 9: $C_9 = 820$
  - 10: $C_{10} = 570$

- Books, values, and weights (in order):
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

**Decision variables:**

For each bookshelf $i \in S$ and each book $j \in B$:
$$
x_{ij} = \text{number of units of book $j$ placed on bookshelf $i$}, \quad x_{ij} \in \mathbb{Z}_{\geq 0}
$$

**Complete Model:**

$$
\begin{align*}
\max \quad & \sum_{i \in S} \sum_{j \in B} v_j \cdot x_{ij} \\
\text{s.t.} \quad & \sum_{j \in B} w_j \cdot x_{ij} \leq C_i, \quad \forall i \in S \\
& x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in S, \forall j \in B
\end{align*}
$$

Where all sets, parameters, and variable definitions are as above, using the exact identifiers and coefficients from the supplied CSV files.