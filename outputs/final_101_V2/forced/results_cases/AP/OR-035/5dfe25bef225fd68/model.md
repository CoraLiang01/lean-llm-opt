##### Decision Variables

Let $x_i$ be the number of units of bread type $i$ to order each day, where $i$ indexes the bread types listed below. All $x_i$ are required to be integers.

##### Objective Function

$\max \left( 888\,x_1 + 134\,x_2 + 129\,x_3 + 370\,x_4 + 921\,x_5 + 765\,x_6 + 154\,x_7 + 837\,x_8 + 584\,x_9 + 365\,x_{10} \right)$

##### Constraints

$\quad 4\,x_1 + 2\,x_2 + 4\,x_3 + 3\,x_4 + 2\,x_5 + 1\,x_6 + 2\,x_7 + 1\,x_8 + 3\,x_9 + 3\,x_{10} \leq 180$

$\quad x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,2,\ldots,10\}$

##### Retrieved Information

{
  "capacity": 180,
  "products": [
    {"ProductName": "Baguette", "Value": 888, "Weight": 4},
    {"ProductName": "Croissant", "Value": 134, "Weight": 2},
    {"ProductName": "Sourdough", "Value": 129, "Weight": 4},
    {"ProductName": "Rye Bread", "Value": 370, "Weight": 3},
    {"ProductName": "Brioche", "Value": 921, "Weight": 2},
    {"ProductName": "Focaccia", "Value": 765, "Weight": 1},
    {"ProductName": "Ciabatta", "Value": 154, "Weight": 2},
    {"ProductName": "Pita", "Value": 837, "Weight": 1},
    {"ProductName": "Bagel", "Value": 584, "Weight": 3},
    {"ProductName": "English Muffin", "Value": 365, "Weight": 3}
  ]
}

##### Variable Index Mapping

| $i$ | Product Name      | Value | Weight |
|-----|------------------|-------|--------|
| 1   | Baguette         | 888   | 4      |
| 2   | Croissant        | 134   | 2      |
| 3   | Sourdough        | 129   | 4      |
| 4   | Rye Bread        | 370   | 3      |
| 5   | Brioche          | 921   | 2      |
| 6   | Focaccia         | 765   | 1      |
| 7   | Ciabatta         | 154   | 2      |
| 8   | Pita             | 837   | 1      |
| 9   | Bagel            | 584   | 3      |
| 10  | English Muffin   | 365   | 3      |