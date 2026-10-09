Let:
- I = {1, 2, ..., 10} be the set of BookshelfIDs (from capacity.csv, in order).
- J = {1, 2, ..., 24} be the set of books, indexed in the order they appear in products.csv.
- Let ProductName_j, Value_j, Weight_j be the name, value, and weight of book j (see below).
- Let Capacity_i be the capacity of bookshelf i (see below).
- Decision variables: x_{ij} = number of units of book j placed on bookshelf i, for all i in I, j in J. All x_{ij} are nonnegative integers.

Data (in supplied order):

Bookshelf capacities (capacity.csv):
| BookshelfID (i) | Capacity_i |
|-----------------|------------|
| 1               | 200        |
| 2               | 200        |
| 3               | 300        |
| 4               | 400        |
| 5               | 550        |
| 6               | 600        |
| 7               | 650        |
| 8               | 750        |
| 9               | 820        |
| 10              | 570        |

Books (products.csv, in order):
| j | ProductName                      | Value_j | Weight_j |
|---|----------------------------------|---------|----------|
| 1 | The Great Gatsby                 | 50      | 10       |
| 2 | To Kill a Mockingbird            | 70      | 20       |
| 3 | 1984                             | 30      | 5        |
| 4 | Pride and Prejudice              | 60      | 15       |
| 5 | The Catcher in the Rye           | 80      | 25       |
| 6 | Moby Dick                        | 90      | 30       |
| 7 | Jane Eyre                        | 40      | 12       |
| 8 | War and Peace                    | 100     | 35       |
| 9 | The Odyssey                      | 55      | 10       |
|10 | Crime and Punishment             | 75      | 20       |
|11 | The Hobbit                       | 65      | 18       |
|12 | Brave New World                  | 95      | 28       |
|13 | Anna Karenina                    | 45      | 8        |
|14 | Wuthering Heights                | 85      | 22       |
|15 | The Divine Comedy                | 70      | 25       |
|16 | The Iliad                        | 110     | 40       |
|17 | Les Misérables                   | 50      | 14       |
|18 | Dracula                          | 60      | 16       |
|19 | Frankenstein                     | 120     | 50       |
|20 | The Brothers Karamazov           | 100     | 30       |
|21 | Don Quixote                      | 52      | 11       |
|22 | One Hundred Years of Solitude    | 68      | 19       |
|23 | Ulysses                          | 38      | 7        |
|24 | The Alchemist                    | 58      | 14       |
|25 | Meditations                      | 82      | 24       |

But only 24 books are listed in the data above (Meditations is the 24th, not 25th), so J = {1,...,24}.

Mathematical Model:

Variables:
x_{ij} ∈ {0, 1, 2, ...} for all i ∈ {1,...,10}, j ∈ {1,...,24}

Objective:
Maximize total value of books placed:
\[
\text{Maximize} \quad Z = \sum_{i=1}^{10} \sum_{j=1}^{24} \text{Value}_j \cdot x_{ij}
\]
where Value_j is as given above for each book j.

Subject to (for each bookshelf i):
\[
\sum_{j=1}^{24} \text{Weight}_j \cdot x_{ij} \leq \text{Capacity}_i \qquad \forall i \in \{1,...,10\}
\]
where Weight_j and Capacity_i are as given above.

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,...,10\},\ j \in \{1,...,24\}
\]

All data is used in the original supplied order, and all bookshelf and book indices are explicit.

Summary:
- Decision variables: x_{ij} = number of units of book j on bookshelf i (integer, ≥0)
- Objective: maximize total value across all shelves
- Constraints: for each shelf, total weight of books ≤ shelf capacity
- All coefficients and indices as above.