##### Sets

Let $I = \{1,2,3,4,5,6,7,8,9,10\}$ be the set of bookshelves (BookshelfID from capacity.csv).

Let $J$ be the set of books (ProductName from products.csv):

\[
J = \{
\text{The Great Gatsby},
\text{To Kill a Mockingbird},
\text{1984},
\text{Pride and Prejudice},
\text{The Catcher in the Rye},
\text{Moby Dick},
\text{Jane Eyre},
\text{War and Peace},
\text{The Odyssey},
\text{Crime and Punishment},
\text{The Hobbit},
\text{Brave New World},
\text{Anna Karenina},
\text{Wuthering Heights},
\text{The Divine Comedy},
\text{The Iliad},
\text{Les Misérables},
\text{Dracula},
\text{Frankenstein},
\text{The Brothers Karamazov},
\text{Don Quixote},
\text{One Hundred Years of Solitude},
\text{Ulysses},
\text{The Alchemist},
\text{Meditations}
\}
\]

##### Parameters

For each bookshelf $i \in I$:

- $C_i$ = capacity of bookshelf $i$.

\[
\begin{align*}
C_1 &= 200 \\
C_2 &= 200 \\
C_3 &= 300 \\
C_4 &= 400 \\
C_5 &= 550 \\
C_6 &= 600 \\
C_7 &= 650 \\
C_8 &= 750 \\
C_9 &= 820 \\
C_{10} &= 570 \\
\end{align*}
\]

For each book $j \in J$:

- $v_j$ = value of book $j$
- $w_j$ = weight of book $j$

\[
\begin{array}{lll}
\text{ProductName} & v_j & w_j \\
\hline
\text{The Great Gatsby} & 50 & 10 \\
\text{To Kill a Mockingbird} & 70 & 20 \\
\text{1984} & 30 & 5 \\
\text{Pride and Prejudice} & 60 & 15 \\
\text{The Catcher in the Rye} & 80 & 25 \\
\text{Moby Dick} & 90 & 30 \\
\text{Jane Eyre} & 40 & 12 \\
\text{War and Peace} & 100 & 35 \\
\text{The Odyssey} & 55 & 10 \\
\text{Crime and Punishment} & 75 & 20 \\
\text{The Hobbit} & 65 & 18 \\
\text{Brave New World} & 95 & 28 \\
\text{Anna Karenina} & 45 & 8 \\
\text{Wuthering Heights} & 85 & 22 \\
\text{The Divine Comedy} & 70 & 25 \\
\text{The Iliad} & 110 & 40 \\
\text{Les Misérables} & 50 & 14 \\
\text{Dracula} & 60 & 16 \\
\text{Frankenstein} & 120 & 50 \\
\text{The Brothers Karamazov} & 100 & 30 \\
\text{Don Quixote} & 52 & 11 \\
\text{One Hundred Years of Solitude} & 68 & 19 \\
\text{Ulysses} & 38 & 7 \\
\text{The Alchemist} & 58 & 14 \\
\text{Meditations} & 82 & 24 \\
\end{array}
\]

##### Decision Variables

For each bookshelf $i \in I$ and book $j \in J$:

- $x_{ij} \in \mathbb{Z}_{\geq 0}$: number of units of book $j$ placed on bookshelf $i$.

##### Objective

\[
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
\]

##### Constraints

For each bookshelf $i \in I$:

\[
\sum_{j \in J} w_j x_{ij} \leq C_i
\]

For all $i \in I$, $j \in J$:

\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]

##### Complete Model

\[
\begin{align*}
\max\quad & \sum_{i=1}^{10} \sum_{j \in J} v_j x_{ij} \\
\text{s.t.}\quad & \sum_{j \in J} w_j x_{ij} \leq C_i \quad \forall i=1,\ldots,10 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i=1,\ldots,10,\ j \in J
\end{align*}
\]

where $C_i$, $v_j$, $w_j$ are as listed above.

###### Retrieved Information

Bookshelf capacities (capacity.csv, in source order):

- BookshelfID 1: 200
- BookshelfID 2: 200
- BookshelfID 3: 300
- BookshelfID 4: 400
- BookshelfID 5: 550
- BookshelfID 6: 600
- BookshelfID 7: 650
- BookshelfID 8: 750
- BookshelfID 9: 820
- BookshelfID 10: 570

Book values and weights (products.csv, in source order):

- The Great Gatsby: Value 50, Weight 10
- To Kill a Mockingbird: Value 70, Weight 20
- 1984: Value 30, Weight 5
- Pride and Prejudice: Value 60, Weight 15
- The Catcher in the Rye: Value 80, Weight 25
- Moby Dick: Value 90, Weight 30
- Jane Eyre: Value 40, Weight 12
- War and Peace: Value 100, Weight 35
- The Odyssey: Value 55, Weight 10
- Crime and Punishment: Value 75, Weight 20
- The Hobbit: Value 65, Weight 18
- Brave New World: Value 95, Weight 28
- Anna Karenina: Value 45, Weight 8
- Wuthering Heights: Value 85, Weight 22
- The Divine Comedy: Value 70, Weight 25
- The Iliad: Value 110, Weight 40
- Les Misérables: Value 50, Weight 14
- Dracula: Value 60, Weight 16
- Frankenstein: Value 120, Weight 50
- The Brothers Karamazov: Value 100, Weight 30
- Don Quixote: Value 52, Weight 11
- One Hundred Years of Solitude: Value 68, Weight 19
- Ulysses: Value 38, Weight 7
- The Alchemist: Value 58, Weight 14
- Meditations: Value 82, Weight 24