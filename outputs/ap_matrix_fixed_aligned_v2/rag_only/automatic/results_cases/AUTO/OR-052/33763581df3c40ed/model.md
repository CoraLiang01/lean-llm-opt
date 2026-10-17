Let:
- \( I = \{1,2,3,4,5,6,7,8,9,10\} \) be the set of BookshelfIDs (from capacity.csv, in order).
- \( J \) be the set of books, indexed in the order from products.csv, with names as below.

Define decision variables:
- \( x_{ij} \): integer, number of units of book \( j \) (ProductName) to place on bookshelf \( i \) (BookshelfID), for all \( i \in I \), \( j \in J \).
- \( x_{ij} \geq 0 \), integer.

Parameters:
- \( c_i \): capacity of bookshelf \( i \) (from capacity.csv).
- \( v_j \): value of book \( j \) (from products.csv).
- \( w_j \): weight of book \( j \) (from products.csv).

Data (in source order):

Bookshelf capacities (capacity.csv):
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

Books (products.csv, in order):
\[
\begin{array}{lll}
j & \text{ProductName} & (\text{Value}, \text{Weight}) \\
1 & \text{The Great Gatsby} & (50, 10) \\
2 & \text{To Kill a Mockingbird} & (70, 20) \\
3 & \text{1984} & (30, 5) \\
4 & \text{Pride and Prejudice} & (60, 15) \\
5 & \text{The Catcher in the Rye} & (80, 25) \\
6 & \text{Moby Dick} & (90, 30) \\
7 & \text{Jane Eyre} & (40, 12) \\
8 & \text{War and Peace} & (100, 35) \\
9 & \text{The Odyssey} & (55, 10) \\
10 & \text{Crime and Punishment} & (75, 20) \\
11 & \text{The Hobbit} & (65, 18) \\
12 & \text{Brave New World} & (95, 28) \\
13 & \text{Anna Karenina} & (45, 8) \\
14 & \text{Wuthering Heights} & (85, 22) \\
15 & \text{The Divine Comedy} & (70, 25) \\
16 & \text{The Iliad} & (110, 40) \\
17 & \text{Les Misérables} & (50, 14) \\
18 & \text{Dracula} & (60, 16) \\
19 & \text{Frankenstein} & (120, 50) \\
20 & \text{The Brothers Karamazov} & (100, 30) \\
21 & \text{Don Quixote} & (52, 11) \\
22 & \text{One Hundred Years of Solitude} & (68, 19) \\
23 & \text{Ulysses} & (38, 7) \\
24 & \text{The Alchemist} & (58, 14) \\
25 & \text{Meditations} & (82, 24) \\
\end{array}
\]

Model:

Maximize total value:
\[
\text{Maximize} \quad Z = \sum_{i=1}^{10} \sum_{j=1}^{25} v_j x_{ij}
\]
where \( v_j \) is as above for each book \( j \).

Subject to, for each bookshelf \( i \) (in source order):

\[
\sum_{j=1}^{25} w_j x_{ij} \leq c_i \qquad \forall i \in \{1,2,\ldots,10\}
\]
where \( w_j \) is as above for each book \( j \), and \( c_i \) is the capacity for shelf \( i \) as above.

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,2,\ldots,10\},\ j \in \{1,2,\ldots,25\}
\]

All coefficients and identifiers are as given in the source data, in original order.

Summary:
- Decision variables: \( x_{ij} \) = number of units of book \( j \) on shelf \( i \), integer, nonnegative.
- Objective: maximize total value of all books placed.
- Constraints: for each shelf, total weight of books placed does not exceed its capacity.
- All data and indices are preserved in source order.