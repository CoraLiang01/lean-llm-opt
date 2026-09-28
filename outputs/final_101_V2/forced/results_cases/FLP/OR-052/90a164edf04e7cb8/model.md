##### Decision Variables

$x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of units of book $j$ to be placed on bookshelf $i$, for each bookshelf $i \in I$ and book $j \in J$.

##### Parameters

- $I = \{1,2,3,4,5,6,7,8,9,10\}$ (Bookshelf IDs)
- $J =$ 
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

- Bookshelf capacities $C_i$:
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

- Book values $v_j$ and weights $w_j$:

| $j$ | Product Name                        | $v_j$ | $w_j$ |
|-----|-------------------------------------|-------|-------|
| 1   | The Great Gatsby                    | 50    | 10    |
| 2   | To Kill a Mockingbird               | 70    | 20    |
| 3   | 1984                                | 30    | 5     |
| 4   | Pride and Prejudice                 | 60    | 15    |
| 5   | The Catcher in the Rye              | 80    | 25    |
| 6   | Moby Dick                           | 90    | 30    |
| 7   | Jane Eyre                           | 40    | 12    |
| 8   | War and Peace                       | 100   | 35    |
| 9   | The Odyssey                         | 55    | 10    |
| 10  | Crime and Punishment                | 75    | 20    |
| 11  | The Hobbit                          | 65    | 18    |
| 12  | Brave New World                     | 95    | 28    |
| 13  | Anna Karenina                       | 45    | 8     |
| 14  | Wuthering Heights                   | 85    | 22    |
| 15  | The Divine Comedy                   | 70    | 25    |
| 16  | The Iliad                           | 110   | 40    |
| 17  | Les Misérables                      | 50    | 14    |
| 18  | Dracula                             | 60    | 16    |
| 19  | Frankenstein                        | 120   | 50    |
| 20  | The Brothers Karamazov              | 100   | 30    |
| 21  | Don Quixote                         | 52    | 11    |
| 22  | One Hundred Years of Solitude       | 68    | 19    |
| 23  | Ulysses                             | 38    | 7     |
| 24  | The Alchemist                       | 58    | 14    |
| 25  | Meditations                         | 82    | 24    |

##### Objective Function

\[
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
\]

##### Constraints

1. **Bookshelf capacity constraints:**
   \[
   \sum_{j \in J} w_j x_{ij} \leq C_i, \quad \forall i \in I
   \]

2. **Integrality:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I,\, j \in J
   \]

##### Complete Model

\[
\begin{align*}
\max \quad & \sum_{i \in I} \sum_{j \in J} v_j x_{ij} \\
\text{s.t.} \quad & \sum_{j \in J} w_j x_{ij} \leq C_i, \quad \forall i \in I \\
& x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I,\, j \in J
\end{align*}
\]

Where all parameters ($C_i$, $v_j$, $w_j$) are as listed above.