Let $x_{ij}$ be the number of units of book $j$ (ProductName) to be placed on bookshelf $i$ (BookshelfID). All $x_{ij}$ are integer and $\geq 0$.

Let $v_j$ be the Value of book $j$, and $w_j$ be the Weight of book $j$.

Let $C_i$ be the Capacity of bookshelf $i$.

Indices:
- $i \in \{1,2,3,4,5,6,7,8,9,10\}$ (BookshelfID)
- $j$ runs over all ProductName entries below.

Objective:
\[
\max \sum_{i=1}^{10} \sum_{j} v_j \cdot x_{ij}
\]

Subject to, for each bookshelf $i$:
\[
\sum_{j} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,2,3,4,5,6,7,8,9,10\}
\]

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
\]

---

#### Data

**Bookshelf Capacities:**

| BookshelfID | Capacity |
|-------------|----------|
| 1           | 200      |
| 2           | 200      |
| 3           | 300      |
| 4           | 400      |
| 5           | 550      |
| 6           | 600      |
| 7           | 650      |
| 8           | 750      |
| 9           | 820      |
| 10          | 570      |

**Books (Products):**

| ProductName                      | Value | Weight |
|-----------------------------------|-------|--------|
| The Great Gatsby                  | 50    | 10     |
| To Kill a Mockingbird             | 70    | 20     |
| 1984                              | 30    | 5      |
| Pride and Prejudice               | 60    | 15     |
| The Catcher in the Rye            | 80    | 25     |
| Moby Dick                         | 90    | 30     |
| Jane Eyre                         | 40    | 12     |
| War and Peace                     | 100   | 35     |
| The Odyssey                       | 55    | 10     |
| Crime and Punishment              | 75    | 20     |
| The Hobbit                        | 65    | 18     |
| Brave New World                   | 95    | 28     |
| Anna Karenina                     | 45    | 8      |
| Wuthering Heights                 | 85    | 22     |
| The Divine Comedy                 | 70    | 25     |
| The Iliad                         | 110   | 40     |
| Les Misérables                    | 50    | 14     |
| Dracula                           | 60    | 16     |
| Frankenstein                      | 120   | 50     |
| The Brothers Karamazov            | 100   | 30     |
| Don Quixote                       | 52    | 11     |
| One Hundred Years of Solitude     | 68    | 19     |
| Ulysses                           | 38    | 7      |
| The Alchemist                     | 58    | 14     |
| Meditations                       | 82    | 24     |

---

**Complete Model:**

\[
\begin{align*}
\max\ & \sum_{i=1}^{10} \Big[ 
50\,x_{i,\text{The Great Gatsby}} + 
70\,x_{i,\text{To Kill a Mockingbird}} + 
30\,x_{i,\text{1984}} + 
60\,x_{i,\text{Pride and Prejudice}} + 
80\,x_{i,\text{The Catcher in the Rye}} + \\
&\quad 90\,x_{i,\text{Moby Dick}} + 
40\,x_{i,\text{Jane Eyre}} + 
100\,x_{i,\text{War and Peace}} + 
55\,x_{i,\text{The Odyssey}} + 
75\,x_{i,\text{Crime and Punishment}} + \\
&\quad 65\,x_{i,\text{The Hobbit}} + 
95\,x_{i,\text{Brave New World}} + 
45\,x_{i,\text{Anna Karenina}} + 
85\,x_{i,\text{Wuthering Heights}} + 
70\,x_{i,\text{The Divine Comedy}} + \\
&\quad 110\,x_{i,\text{The Iliad}} + 
50\,x_{i,\text{Les Misérables}} + 
60\,x_{i,\text{Dracula}} + 
120\,x_{i,\text{Frankenstein}} + 
100\,x_{i,\text{The Brothers Karamazov}} + \\
&\quad 52\,x_{i,\text{Don Quixote}} + 
68\,x_{i,\text{One Hundred Years of Solitude}} + 
38\,x_{i,\text{Ulysses}} + 
58\,x_{i,\text{The Alchemist}} + 
82\,x_{i,\text{Meditations}}
\Big]
\end{align*}
\]

Subject to, for each $i$:

\[
\begin{align*}
10\,x_{i,\text{The Great Gatsby}} + 
20\,x_{i,\text{To Kill a Mockingbird}} + 
5\,x_{i,\text{1984}} + 
15\,x_{i,\text{Pride and Prejudice}} + 
25\,x_{i,\text{The Catcher in the Rye}} + \\
30\,x_{i,\text{Moby Dick}} + 
12\,x_{i,\text{Jane Eyre}} + 
35\,x_{i,\text{War and Peace}} + 
10\,x_{i,\text{The Odyssey}} + 
20\,x_{i,\text{Crime and Punishment}} + \\
18\,x_{i,\text{The Hobbit}} + 
28\,x_{i,\text{Brave New World}} + 
8\,x_{i,\text{Anna Karenina}} + 
22\,x_{i,\text{Wuthering Heights}} + 
25\,x_{i,\text{The Divine Comedy}} + \\
40\,x_{i,\text{The Iliad}} + 
14\,x_{i,\text{Les Misérables}} + 
16\,x_{i,\text{Dracula}} + 
50\,x_{i,\text{Frankenstein}} + 
30\,x_{i,\text{The Brothers Karamazov}} + \\
11\,x_{i,\text{Don Quixote}} + 
19\,x_{i,\text{One Hundred Years of Solitude}} + 
7\,x_{i,\text{Ulysses}} + 
14\,x_{i,\text{The Alchemist}} + 
24\,x_{i,\text{Meditations}}
\leq C_i
\end{align*}
\]
where $C_i$ is the capacity for bookshelf $i$ as given above.

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
\]