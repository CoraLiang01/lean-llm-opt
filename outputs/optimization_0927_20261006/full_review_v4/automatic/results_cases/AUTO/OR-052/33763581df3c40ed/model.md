Let $x_{ij}$ be the number of units of book $j$ (ProductName) placed on bookshelf $i$ (BookshelfID). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- Bookshelves (from capacity.csv, in order):

  1. BookshelfID = 1, Capacity = 200
  2. BookshelfID = 2, Capacity = 200
  3. BookshelfID = 3, Capacity = 300
  4. BookshelfID = 4, Capacity = 400
  5. BookshelfID = 5, Capacity = 550
  6. BookshelfID = 6, Capacity = 600
  7. BookshelfID = 7, Capacity = 650
  8. BookshelfID = 8, Capacity = 750
  9. BookshelfID = 9, Capacity = 820
  10. BookshelfID = 10, Capacity = 570

- Books (from products.csv, in order):

  1. The Great Gatsby, Value = 50, Weight = 10
  2. To Kill a Mockingbird, Value = 70, Weight = 20
  3. 1984, Value = 30, Weight = 5
  4. Pride and Prejudice, Value = 60, Weight = 15
  5. The Catcher in the Rye, Value = 80, Weight = 25
  6. Moby Dick, Value = 90, Weight = 30
  7. Jane Eyre, Value = 40, Weight = 12
  8. War and Peace, Value = 100, Weight = 35
  9. The Odyssey, Value = 55, Weight = 10
  10. Crime and Punishment, Value = 75, Weight = 20
  11. The Hobbit, Value = 65, Weight = 18
  12. Brave New World, Value = 95, Weight = 28
  13. Anna Karenina, Value = 45, Weight = 8
  14. Wuthering Heights, Value = 85, Weight = 22
  15. The Divine Comedy, Value = 70, Weight = 25
  16. The Iliad, Value = 110, Weight = 40
  17. Les Misérables, Value = 50, Weight = 14
  18. Dracula, Value = 60, Weight = 16
  19. Frankenstein, Value = 120, Weight = 50
  20. The Brothers Karamazov, Value = 100, Weight = 30
  21. Don Quixote, Value = 52, Weight = 11
  22. One Hundred Years of Solitude, Value = 68, Weight = 19
  23. Ulysses, Value = 38, Weight = 7
  24. The Alchemist, Value = 58, Weight = 14
  25. Meditations, Value = 82, Weight = 24

---

### Mathematical Model

**Decision Variables:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,25\}
$$

where $x_{ij}$ is the number of units of book $j$ placed on bookshelf $i$.

---

**Objective:**

$$
\max \sum_{i=1}^{10} \sum_{j=1}^{25} v_j \cdot x_{ij}
$$

where $v_j$ is the value of book $j$ (see list above).

---

**Constraints:**

For each bookshelf $i$ (BookshelfID), the total weight of books placed cannot exceed its capacity:

$$
\sum_{j=1}^{25} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,10\}
$$

where $w_j$ is the weight of book $j$ and $C_i$ is the capacity of bookshelf $i$ (see list above).

---

**Variable Domains:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i,j
$$

---

**Parameter Table (Books):**

| $j$ | ProductName                      | $v_j$ (Value) | $w_j$ (Weight) |
|-----|----------------------------------|---------------|---------------|
| 1   | The Great Gatsby                 | 50            | 10            |
| 2   | To Kill a Mockingbird            | 70            | 20            |
| 3   | 1984                             | 30            | 5             |
| 4   | Pride and Prejudice              | 60            | 15            |
| 5   | The Catcher in the Rye           | 80            | 25            |
| 6   | Moby Dick                        | 90            | 30            |
| 7   | Jane Eyre                        | 40            | 12            |
| 8   | War and Peace                    | 100           | 35            |
| 9   | The Odyssey                      | 55            | 10            |
| 10  | Crime and Punishment             | 75            | 20            |
| 11  | The Hobbit                       | 65            | 18            |
| 12  | Brave New World                  | 95            | 28            |
| 13  | Anna Karenina                    | 45            | 8             |
| 14  | Wuthering Heights                | 85            | 22            |
| 15  | The Divine Comedy                | 70            | 25            |
| 16  | The Iliad                        | 110           | 40            |
| 17  | Les Misérables                   | 50            | 14            |
| 18  | Dracula                          | 60            | 16            |
| 19  | Frankenstein                     | 120           | 50            |
| 20  | The Brothers Karamazov           | 100           | 30            |
| 21  | Don Quixote                      | 52            | 11            |
| 22  | One Hundred Years of Solitude    | 68            | 19            |
| 23  | Ulysses                          | 38            | 7             |
| 24  | The Alchemist                    | 58            | 14            |
| 25  | Meditations                      | 82            | 24            |

**Parameter Table (Bookshelves):**

| $i$ | BookshelfID | $C_i$ (Capacity) |
|-----|-------------|------------------|
| 1   | 1           | 200              |
| 2   | 2           | 200              |
| 3   | 3           | 300              |
| 4   | 4           | 400              |
| 5   | 5           | 550              |
| 6   | 6           | 600              |
| 7   | 7           | 650              |
| 8   | 8           | 750              |
| 9   | 9           | 820              |
| 10  | 10          | 570              |