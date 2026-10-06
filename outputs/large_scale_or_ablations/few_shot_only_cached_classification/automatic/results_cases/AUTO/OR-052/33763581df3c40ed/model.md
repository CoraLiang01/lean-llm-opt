Let $x_{ij}$ be the number of units of book $j$ to be placed on bookshelf $i$, where $i$ indexes BookshelfID and $j$ indexes ProductName. All $x_{ij}$ are integer and $\geq 0$.

**Parameters:**

- Bookshelves ($i$):  
  1, 2, 3, 4, 5, 6, 7, 8, 9, 10

- Books ($j$) with Value $v_j$ and Weight $w_j$:

| ProductName                      | Value ($v_j$) | Weight ($w_j$) |
|----------------------------------|:-------------:|:--------------:|
| The Great Gatsby                 |      50       |      10        |
| To Kill a Mockingbird            |      70       |      20        |
| 1984                             |      30       |       5        |
| Pride and Prejudice              |      60       |      15        |
| The Catcher in the Rye           |      80       |      25        |
| Moby Dick                        |      90       |      30        |
| Jane Eyre                        |      40       |      12        |
| War and Peace                    |     100       |      35        |
| The Odyssey                      |      55       |      10        |
| Crime and Punishment             |      75       |      20        |
| The Hobbit                       |      65       |      18        |
| Brave New World                  |      95       |      28        |
| Anna Karenina                    |      45       |       8        |
| Wuthering Heights                |      85       |      22        |
| The Divine Comedy                |      70       |      25        |
| The Iliad                        |     110       |      40        |
| Les Misérables                   |      50       |      14        |
| Dracula                          |      60       |      16        |
| Frankenstein                     |     120       |      50        |
| The Brothers Karamazov           |     100       |      30        |
| Don Quixote                      |      52       |      11        |
| One Hundred Years of Solitude    |      68       |      19        |
| Ulysses                          |      38       |       7        |
| The Alchemist                    |      58       |      14        |
| Meditations                      |      82       |      24        |

- Bookshelf capacities ($C_i$):

| BookshelfID | Capacity ($C_i$) |
|:-----------:|:----------------:|
|      1      |       200        |
|      2      |       200        |
|      3      |       300        |
|      4      |       400        |
|      5      |       550        |
|      6      |       600        |
|      7      |       650        |
|      8      |       750        |
|      9      |       820        |
|     10      |       570        |

---

**Mathematical Model:**

**Objective:**
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{25} v_j \cdot x_{ij}
\]

**Subject to:**

For each bookshelf $i$ (BookshelfID $i$):

\[
\sum_{j=1}^{25} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,2,\ldots,10\}
\]

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,2,\ldots,10\},\; j \in \{1,2,\ldots,25\}
\]

Where:

- $v_j$ and $w_j$ are the Value and Weight of book $j$ as listed above.
- $C_i$ is the Capacity of bookshelf $i$ as listed above.
- $x_{ij}$ is the integer number of units of book $j$ placed on bookshelf $i$.