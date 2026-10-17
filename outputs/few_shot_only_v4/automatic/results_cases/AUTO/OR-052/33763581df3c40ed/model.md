Let $x_{ij}$ be the number of units of book $j$ to be placed on bookshelf $i$.

Indices:
- $i \in \{1,2,3,4,5,6,7,8,9,10\}$ (BookshelfID)
- $j \in \{$The Great Gatsby, To Kill a Mockingbird, 1984, Pride and Prejudice, The Catcher in the Rye, Moby Dick, Jane Eyre, War and Peace, The Odyssey, Crime and Punishment, The Hobbit, Brave New World, Anna Karenina, Wuthering Heights, The Divine Comedy, The Iliad, Les Misérables, Dracula, Frankenstein, The Brothers Karamazov, Don Quixote, One Hundred Years of Solitude, Ulysses, The Alchemist, Meditations$\}$ (ProductName)

Parameters:
- $v_j$ = Value of book $j$ (from products.csv)
- $w_j$ = Weight of book $j$ (from products.csv)
- $C_i$ = Capacity of bookshelf $i$ (from capacity.csv)

Data:

Bookshelves and Capacities:
\[
\begin{array}{ll}
\text{BookshelfID} & \text{Capacity} \\
1 & 200 \\
2 & 200 \\
3 & 300 \\
4 & 400 \\
5 & 550 \\
6 & 600 \\
7 & 650 \\
8 & 750 \\
9 & 820 \\
10 & 570 \\
\end{array}
\]

Books, Values, and Weights:
\[
\begin{array}{lll}
\text{ProductName} & \text{Value} & \text{Weight} \\
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

Objective:
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{25} v_j \cdot x_{ij}
\]

Subject to:

For each bookshelf $i$:
\[
\sum_{j=1}^{25} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,2,\ldots,10\}
\]

Integrality and nonnegativity:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,2,\ldots,10\},\; \forall j \in \{1,2,\ldots,25\}
\]

Where:
- $v_j$ and $w_j$ are as given above for each ProductName $j$.
- $C_i$ is as given above for each BookshelfID $i$.