##### Sets

- $I$ = set of bookshelves, indexed by $i$ (BookshelfID: 1, 2, ..., 10)
- $J$ = set of books, indexed by $j$ (see ProductName list below)

##### Parameters

- $C_i$ = capacity of bookshelf $i$

  $C_1 = 200$  
  $C_2 = 200$  
  $C_3 = 300$  
  $C_4 = 400$  
  $C_5 = 550$  
  $C_6 = 600$  
  $C_7 = 650$  
  $C_8 = 750$  
  $C_9 = 820$  
  $C_{10} = 570$

- $v_j$ = value of one unit of book $j$
- $w_j$ = weight of one unit of book $j$

| $j$ | ProductName                        | $v_j$ | $w_j$ |
|-----|------------------------------------|-------|-------|
| 1   | The Great Gatsby                   | 50    | 10    |
| 2   | To Kill a Mockingbird              | 70    | 20    |
| 3   | 1984                               | 30    | 5     |
| 4   | Pride and Prejudice                | 60    | 15    |
| 5   | The Catcher in the Rye             | 80    | 25    |
| 6   | Moby Dick                          | 90    | 30    |
| 7   | Jane Eyre                          | 40    | 12    |
| 8   | War and Peace                      | 100   | 35    |
| 9   | The Odyssey                        | 55    | 10    |
| 10  | Crime and Punishment               | 75    | 20    |
| 11  | The Hobbit                         | 65    | 18    |
| 12  | Brave New World                    | 95    | 28    |
| 13  | Anna Karenina                      | 45    | 8     |
| 14  | Wuthering Heights                  | 85    | 22    |
| 15  | The Divine Comedy                  | 70    | 25    |
| 16  | The Iliad                          | 110   | 40    |
| 17  | Les Misérables                     | 50    | 14    |
| 18  | Dracula                            | 60    | 16    |
| 19  | Frankenstein                       | 120   | 50    |
| 20  | The Brothers Karamazov             | 100   | 30    |
| 21  | Don Quixote                        | 52    | 11    |
| 22  | One Hundred Years of Solitude      | 68    | 19    |
| 23  | Ulysses                            | 38    | 7     |
| 24  | The Alchemist                      | 58    | 14    |
| 25  | Meditations                        | 82    | 24    |

##### Decision Variables

- $x_{ij}$ = number of units of book $j$ placed on bookshelf $i$, integer, $x_{ij} \geq 0$

##### Objective Function

$\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}$

##### Constraints

1. **Capacity constraints for each bookshelf:**

$\sum_{j \in J} w_j \, x_{ij} \leq C_i \quad \forall i \in I$

2. **Non-negativity and integrality:**

$x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I, \forall j \in J$

##### Retrieved Information

{
  "bookshelves": {
    "1": 200,
    "2": 200,
    "3": 300,
    "4": 400,
    "5": 550,
    "6": 600,
    "7": 650,
    "8": 750,
    "9": 820,
    "10": 570
  },
  "books": [
    {"ProductName": "The Great Gatsby", "Value": 50, "Weight": 10},
    {"ProductName": "To Kill a Mockingbird", "Value": 70, "Weight": 20},
    {"ProductName": "1984", "Value": 30, "Weight": 5},
    {"ProductName": "Pride and Prejudice", "Value": 60, "Weight": 15},
    {"ProductName": "The Catcher in the Rye", "Value": 80, "Weight": 25},
    {"ProductName": "Moby Dick", "Value": 90, "Weight": 30},
    {"ProductName": "Jane Eyre", "Value": 40, "Weight": 12},
    {"ProductName": "War and Peace", "Value": 100, "Weight": 35},
    {"ProductName": "The Odyssey", "Value": 55, "Weight": 10},
    {"ProductName": "Crime and Punishment", "Value": 75, "Weight": 20},
    {"ProductName": "The Hobbit", "Value": 65, "Weight": 18},
    {"ProductName": "Brave New World", "Value": 95, "Weight": 28},
    {"ProductName": "Anna Karenina", "Value": 45, "Weight": 8},
    {"ProductName": "Wuthering Heights", "Value": 85, "Weight": 22},
    {"ProductName": "The Divine Comedy", "Value": 70, "Weight": 25},
    {"ProductName": "The Iliad", "Value": 110, "Weight": 40},
    {"ProductName": "Les Misérables", "Value": 50, "Weight": 14},
    {"ProductName": "Dracula", "Value": 60, "Weight": 16},
    {"ProductName": "Frankenstein", "Value": 120, "Weight": 50},
    {"ProductName": "The Brothers Karamazov", "Value": 100, "Weight": 30},
    {"ProductName": "Don Quixote", "Value": 52, "Weight": 11},
    {"ProductName": "One Hundred Years of Solitude", "Value": 68, "Weight": 19},
    {"ProductName": "Ulysses", "Value": 38, "Weight": 7},
    {"ProductName": "The Alchemist", "Value": 58, "Weight": 14},
    {"ProductName": "Meditations", "Value": 82, "Weight": 24}
  ]
}