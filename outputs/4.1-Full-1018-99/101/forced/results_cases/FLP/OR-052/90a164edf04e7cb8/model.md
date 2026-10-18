##### Sets and Indices

Let $I = \{1,2,3,4,5,6,7,8,9,10\}$ be the set of bookshelves, indexed by $i$ (BookshelfID).
Let $J$ be the set of books, indexed by $j$ (ProductName):

$J = \{$
The Great Gatsby,
To Kill a Mockingbird,
1984,
Pride and Prejudice,
The Catcher in the Rye,
Moby Dick,
Jane Eyre,
War and Peace,
The Odyssey,
Crime and Punishment,
The Hobbit,
Brave New World,
Anna Karenina,
Wuthering Heights,
The Divine Comedy,
The Iliad,
Les Misérables,
Dracula,
Frankenstein,
The Brothers Karamazov,
Don Quixote,
One Hundred Years of Solitude,
Ulysses,
The Alchemist,
Meditations
$\}$

##### Parameters

Bookshelf capacities:
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

Book values and weights:
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

$x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of units of book $j$ placed on bookshelf $i$.

##### Objective Function

\[
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
\]

##### Constraints

1. **Bookshelf capacity constraints** (for each bookshelf $i$):
   \[
   \sum_{j \in J} w_j x_{ij} \leq C_i, \quad \forall i \in I
   \]

2. **Nonnegativity and integrality**:
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I,\, j \in J
   \]

---

###### Retrieved Information

Bookshelves and capacities:
- 1: 200
- 2: 200
- 3: 300
- 4: 400
- 5: 550
- 6: 600
- 7: 650
- 8: 750
- 9: 820
- 10: 570

Books, values, and weights:
- The Great Gatsby: value 50, weight 10
- To Kill a Mockingbird: value 70, weight 20
- 1984: value 30, weight 5
- Pride and Prejudice: value 60, weight 15
- The Catcher in the Rye: value 80, weight 25
- Moby Dick: value 90, weight 30
- Jane Eyre: value 40, weight 12
- War and Peace: value 100, weight 35
- The Odyssey: value 55, weight 10
- Crime and Punishment: value 75, weight 20
- The Hobbit: value 65, weight 18
- Brave New World: value 95, weight 28
- Anna Karenina: value 45, weight 8
- Wuthering Heights: value 85, weight 22
- The Divine Comedy: value 70, weight 25
- The Iliad: value 110, weight 40
- Les Misérables: value 50, weight 14
- Dracula: value 60, weight 16
- Frankenstein: value 120, weight 50
- The Brothers Karamazov: value 100, weight 30
- Don Quixote: value 52, weight 11
- One Hundred Years of Solitude: value 68, weight 19
- Ulysses: value 38, weight 7
- The Alchemist: value 58, weight 14
- Meditations: value 82, weight 24