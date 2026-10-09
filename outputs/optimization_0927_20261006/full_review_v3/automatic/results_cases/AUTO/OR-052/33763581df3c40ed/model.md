Let $x_{ij}$ be the number of units of book $j$ (ProductName) to be placed on bookshelf $i$ (BookshelfID). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- $V_j$: Value of book $j$ (from "Value" in products.csv)
- $W_j$: Weight of book $j$ (from "Weight" in products.csv)
- $C_i$: Capacity of bookshelf $i$ (from "Capacity" in capacity.csv)

**Sets:**

- $i \in \{\text{1}, \text{2}, \text{3}, \text{4}, \text{5}, \text{6}, \text{7}, \text{8}, \text{9}, \text{10}\}$ (BookshelfID, in source order)
- $j \in$ (ProductName, in source order):

  1. The Great Gatsby
  2. To Kill a Mockingbird
  3. 1984
  4. Pride and Prejudice
  5. The Catcher in the Rye
  6. Moby Dick
  7. Jane Eyre
  8. War and Peace
  9. The Odyssey
  10. Crime and Punishment
  11. The Hobbit
  12. Brave New World
  13. Anna Karenina
  14. Wuthering Heights
  15. The Divine Comedy
  16. The Iliad
  17. Les Misérables
  18. Dracula
  19. Frankenstein
  20. The Brothers Karamazov
  21. Don Quixote
  22. One Hundred Years of Solitude
  23. Ulysses
  24. The Alchemist
  25. Meditations

**Objective:**

\[
\max \sum_{i \in \{1,\ldots,10\}} \sum_{j \in \text{Products}} V_j \cdot x_{ij}
\]

where $V_j$ is as follows (in source order):

- The Great Gatsby: 50
- To Kill a Mockingbird: 70
- 1984: 30
- Pride and Prejudice: 60
- The Catcher in the Rye: 80
- Moby Dick: 90
- Jane Eyre: 40
- War and Peace: 100
- The Odyssey: 55
- Crime and Punishment: 75
- The Hobbit: 65
- Brave New World: 95
- Anna Karenina: 45
- Wuthering Heights: 85
- The Divine Comedy: 70
- The Iliad: 110
- Les Misérables: 50
- Dracula: 60
- Frankenstein: 120
- The Brothers Karamazov: 100
- Don Quixote: 52
- One Hundred Years of Solitude: 68
- Ulysses: 38
- The Alchemist: 58
- Meditations: 82

**Constraints:**

For each bookshelf $i$ (BookshelfID, in source order):

- Bookshelf 1: $\sum_{j} W_j x_{1j} \leq 200$
- Bookshelf 2: $\sum_{j} W_j x_{2j} \leq 200$
- Bookshelf 3: $\sum_{j} W_j x_{3j} \leq 300$
- Bookshelf 4: $\sum_{j} W_j x_{4j} \leq 400$
- Bookshelf 5: $\sum_{j} W_j x_{5j} \leq 550$
- Bookshelf 6: $\sum_{j} W_j x_{6j} \leq 600$
- Bookshelf 7: $\sum_{j} W_j x_{7j} \leq 650$
- Bookshelf 8: $\sum_{j} W_j x_{8j} \leq 750$
- Bookshelf 9: $\sum_{j} W_j x_{9j} \leq 820$
- Bookshelf 10: $\sum_{j} W_j x_{10j} \leq 570$

where $W_j$ is as follows (in source order):

- The Great Gatsby: 10
- To Kill a Mockingbird: 20
- 1984: 5
- Pride and Prejudice: 15
- The Catcher in the Rye: 25
- Moby Dick: 30
- Jane Eyre: 12
- War and Peace: 35
- The Odyssey: 10
- Crime and Punishment: 20
- The Hobbit: 18
- Brave New World: 28
- Anna Karenina: 8
- Wuthering Heights: 22
- The Divine Comedy: 25
- The Iliad: 40
- Les Misérables: 14
- Dracula: 16
- Frankenstein: 50
- The Brothers Karamazov: 30
- Don Quixote: 11
- One Hundred Years of Solitude: 19
- Ulysses: 7
- The Alchemist: 14
- Meditations: 24

**Variable domains:**

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,10\},\ j \in \text{Products}
\]

**Complete Model:**

\[
\begin{align*}
\max\ & \sum_{i=1}^{10} \sum_{j=1}^{25} V_j x_{ij} \\
\text{s.t.}\quad
& \sum_{j=1}^{25} W_j x_{ij} \leq C_i, \quad \forall i=1,\ldots,10 \\
& x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i=1,\ldots,10,\ j=1,\ldots,25
\end{align*}
\]

with all $V_j$, $W_j$, and $C_i$ as listed above, and all indices and identifiers preserved in source order.