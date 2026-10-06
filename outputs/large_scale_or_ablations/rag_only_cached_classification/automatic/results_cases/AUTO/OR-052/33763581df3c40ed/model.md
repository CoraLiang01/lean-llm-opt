Let:
- i index the bookshelves, with BookshelfID from capacity.csv: i ∈ {1,2,3,4,5,6,7,8,9,10}
- j index the books, with ProductName from products.csv, in the given order: j ∈ {1,...,25}, with mapping as below.

Define decision variables:
x_{ij} = number of units of book j placed on bookshelf i, for all i and j.
Domain: x_{ij} ∈ {0,1,2,...} (nonnegative integers)

Data:
From capacity.csv:

| BookshelfID (i) | Capacity (C_i) |
|-----------------|---------------|
| 1               | 200           |
| 2               | 200           |
| 3               | 300           |
| 4               | 400           |
| 5               | 550           |
| 6               | 600           |
| 7               | 650           |
| 8               | 750           |
| 9               | 820           |
| 10              | 570           |

From products.csv (j = 1 to 25, in order):

| j | ProductName                     | Value (v_j) | Weight (w_j) |
|---|---------------------------------|-------------|--------------|
| 1 | The Great Gatsby                | 50          | 10           |
| 2 | To Kill a Mockingbird           | 70          | 20           |
| 3 | 1984                            | 30          | 5            |
| 4 | Pride and Prejudice             | 60          | 15           |
| 5 | The Catcher in the Rye          | 80          | 25           |
| 6 | Moby Dick                       | 90          | 30           |
| 7 | Jane Eyre                       | 40          | 12           |
| 8 | War and Peace                   | 100         | 35           |
| 9 | The Odyssey                     | 55          | 10           |
|10 | Crime and Punishment            | 75          | 20           |
|11 | The Hobbit                      | 65          | 18           |
|12 | Brave New World                 | 95          | 28           |
|13 | Anna Karenina                   | 45          | 8            |
|14 | Wuthering Heights               | 85          | 22           |
|15 | The Divine Comedy               | 70          | 25           |
|16 | The Iliad                       | 110         | 40           |
|17 | Les Misérables                  | 50          | 14           |
|18 | Dracula                         | 60          | 16           |
|19 | Frankenstein                    | 120         | 50           |
|20 | The Brothers Karamazov          | 100         | 30           |
|21 | Don Quixote                     | 52          | 11           |
|22 | One Hundred Years of Solitude   | 68          | 19           |
|23 | Ulysses                         | 38          | 7            |
|24 | The Alchemist                   | 58          | 14           |
|25 | Meditations                     | 82          | 24           |

Mathematical Model:

Variables:
x_{ij} ∈ {0,1,2,...} for all i ∈ {1,...,10}, j ∈ {1,...,25}

Objective:
Maximize total value of books placed on all shelves:
\[
\text{Maximize} \quad Z = \sum_{i=1}^{10} \sum_{j=1}^{25} v_j \cdot x_{ij}
\]
where v_j is the Value from products.csv for book j.

Constraints:
For each bookshelf i (BookshelfID from capacity.csv), the total weight of books placed on that shelf cannot exceed its capacity:
\[
\sum_{j=1}^{25} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,...,10\}
\]
where w_j is the Weight from products.csv for book j, and C_i is the Capacity from capacity.csv for bookshelf i.

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,...,10\},\ j \in \{1,...,25\}
\]

All coefficients and identifiers are as given in the CSVs above. No additional constraints or data are assumed.

This is a complete integer programming formulation for the described bookstore allocation problem.