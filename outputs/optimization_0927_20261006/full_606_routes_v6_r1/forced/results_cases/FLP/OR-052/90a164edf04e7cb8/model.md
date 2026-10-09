##### Decision Variables

$x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of units of book $j \in J$ to be placed on bookshelf $i \in I$.

##### Parameters

- $I = \{1,2,3,4,5,6,7,8,9,10\}$ (Bookshelf IDs)
- $J =$ 
  - The Great Gatsby
  - To Kill a Mockingbird
  - 1984
  - Pride and Prejudice
  - The Catcher in the Rye
  - Moby Dick
  - Jane Eyre
  - War and Peace
  - The Odyssey
  - Crime and Punishment
  - The Hobbit
  - Brave New World
  - Anna Karenina
  - Wuthering Heights
  - The Divine Comedy
  - The Iliad
  - Les Misérables
  - Dracula
  - Frankenstein
  - The Brothers Karamazov
  - Don Quixote
  - One Hundred Years of Solitude
  - Ulysses
  - The Alchemist
  - Meditations

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

| Book                          | $v_j$ | $w_j$ |
|-------------------------------|-------|-------|
| The Great Gatsby              | 50    | 10    |
| To Kill a Mockingbird         | 70    | 20    |
| 1984                          | 30    | 5     |
| Pride and Prejudice           | 60    | 15    |
| The Catcher in the Rye        | 80    | 25    |
| Moby Dick                     | 90    | 30    |
| Jane Eyre                     | 40    | 12    |
| War and Peace                 | 100   | 35    |
| The Odyssey                   | 55    | 10    |
| Crime and Punishment          | 75    | 20    |
| The Hobbit                    | 65    | 18    |
| Brave New World               | 95    | 28    |
| Anna Karenina                 | 45    | 8     |
| Wuthering Heights             | 85    | 22    |
| The Divine Comedy             | 70    | 25    |
| The Iliad                     | 110   | 40    |
| Les Misérables                | 50    | 14    |
| Dracula                       | 60    | 16    |
| Frankenstein                  | 120   | 50    |
| The Brothers Karamazov        | 100   | 30    |
| Don Quixote                   | 52    | 11    |
| One Hundred Years of Solitude | 68    | 19    |
| Ulysses                       | 38    | 7     |
| The Alchemist                 | 58    | 14    |
| Meditations                   | 82    | 24    |

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

##### Complete Model

\[
\begin{align*}
\max\quad & \sum_{i \in I} \sum_{j \in J} v_j x_{ij} \\
\text{s.t.}\quad & \sum_{j \in J} w_j x_{ij} \leq C_i, \quad \forall i \in I \\
& x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I,\, j \in J
\end{align*}
\]

Where all parameters ($C_i$, $v_j$, $w_j$) are as listed above.