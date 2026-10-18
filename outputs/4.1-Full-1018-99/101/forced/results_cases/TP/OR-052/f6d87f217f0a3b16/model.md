Let $I = \{1,2,3,4,5,6,7,8,9,10\}$ be the set of bookshelves, and $J$ be the set of books:

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

Let $x_{ij} \in \mathbb{Z}_{\geq 0}$ be the number of units of book $j$ placed on bookshelf $i$.

Parameters:

Bookshelf capacities (from capacity.csv):

\[
\begin{align*}
\text{Capacity}_1 &= 200 \\
\text{Capacity}_2 &= 200 \\
\text{Capacity}_3 &= 300 \\
\text{Capacity}_4 &= 400 \\
\text{Capacity}_5 &= 550 \\
\text{Capacity}_6 &= 600 \\
\text{Capacity}_7 &= 650 \\
\text{Capacity}_8 &= 750 \\
\text{Capacity}_9 &= 820 \\
\text{Capacity}_{10} &= 570 \\
\end{align*}
\]

Book values and weights (from products.csv):

\[
\begin{array}{lll}
\text{Book} & \text{Value} & \text{Weight} \\
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

Model:

Maximize total value:
\[
\max \sum_{i=1}^{10} \sum_{j \in J} v_j x_{ij}
\]
where $v_j$ is the value of book $j$.

Subject to, for each bookshelf $i$:
\[
\sum_{j \in J} w_j x_{ij} \leq \text{Capacity}_i \qquad \forall i=1,\ldots,10
\]
where $w_j$ is the weight of book $j$.

And
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i=1,\ldots,10,\; j \in J
\]

All data and identifiers are preserved as retrieved.

Retrieved Information:

Bookshelf capacities:
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

Books (name, value, weight):
- The Great Gatsby: 50, 10
- To Kill a Mockingbird: 70, 20
- 1984: 30, 5
- Pride and Prejudice: 60, 15
- The Catcher in the Rye: 80, 25
- Moby Dick: 90, 30
- Jane Eyre: 40, 12
- War and Peace: 100, 35
- The Odyssey: 55, 10
- Crime and Punishment: 75, 20
- The Hobbit: 65, 18
- Brave New World: 95, 28
- Anna Karenina: 45, 8
- Wuthering Heights: 85, 22
- The Divine Comedy: 70, 25
- The Iliad: 110, 40
- Les Misérables: 50, 14
- Dracula: 60, 16
- Frankenstein: 120, 50
- The Brothers Karamazov: 100, 30
- Don Quixote: 52, 11
- One Hundred Years of Solitude: 68, 19
- Ulysses: 38, 7
- The Alchemist: 58, 14
- Meditations: 82, 24