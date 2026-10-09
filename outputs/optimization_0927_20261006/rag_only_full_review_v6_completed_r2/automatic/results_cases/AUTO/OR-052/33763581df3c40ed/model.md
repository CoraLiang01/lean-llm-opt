Let:
- \( I \) be the set of BookshelfIDs from capacity.csv: {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
- \( J \) be the set of ProductNames from products.csv (in the given order):

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

Parameters:
- \( C_i \): Capacity of bookshelf \( i \) (from capacity.csv)
- \( v_j \): Value of book \( j \) (from products.csv)
- \( w_j \): Weight of book \( j \) (from products.csv)

Decision variables:
- \( x_{ij} \): Number of units of book \( j \) to place on bookshelf \( i \), integer, \( x_{ij} \geq 0 \)

Data (in source order):

From capacity.csv:
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

From products.csv:
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

Maximize total value:
\[
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
\]

Subject to, for each bookshelf \( i \in I \):
\[
\sum_{j \in J} w_j x_{ij} \leq C_i
\]

And for all \( i \in I, j \in J \):
\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]

Where:
- \( C_i \) is the capacity for bookshelf \( i \) as listed above.
- \( v_j \) and \( w_j \) are the value and weight for book \( j \) as listed above, in the given order.
- All variables and constraints are indexed in the original file order.

This model determines the optimal integer allocation of each book to each bookshelf to maximize total value, subject to the bookshelf weight capacities.