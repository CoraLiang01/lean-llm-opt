##### Decision Variables

Let $x_i$ be the integer number of units of product $i$ to order daily, for each $i$ in the set of products:
- $x_{\text{Spinach}}$
- $x_{\text{Shiitake Mushrooms}}$
- $x_{\text{Apples}}$
- $x_{\text{Carrots}}$
- $x_{\text{Basil}}$
- $x_{\text{Potatoes}}$
- $x_{\text{Green Beans}}$
- $x_{\text{Blueberries}}$
- $x_{\text{Oranges}}$
- $x_{\text{Watermelons}}$

All $x_i \in \mathbb{Z}_{\geq 0}$ (nonnegative integers).

##### Objective Function

$\max\ 49x_{\text{Spinach}} + 30x_{\text{Shiitake Mushrooms}} + 30x_{\text{Apples}} + 18x_{\text{Carrots}} + 54x_{\text{Basil}} + 27x_{\text{Potatoes}} + 91x_{\text{Green Beans}} + 88x_{\text{Blueberries}} + 78x_{\text{Oranges}} + 22x_{\text{Watermelons}}$

##### Constraints

$282x_{\text{Spinach}} + 83x_{\text{Shiitake Mushrooms}} + 251x_{\text{Apples}} + 257x_{\text{Carrots}} + 88x_{\text{Basil}} + 52x_{\text{Potatoes}} + 198x_{\text{Green Beans}} + 203x_{\text{Blueberries}} + 87x_{\text{Oranges}} + 265x_{\text{Watermelons}} \leq 1035$

$x_i \in \mathbb{Z}_{\geq 0}$ for all $i$ (all variables are nonnegative integers).

##### Retrieved Information

- Capacity: $1035$
- Products (in source order):

| ProductName           | Weight | Value |
|-----------------------|--------|-------|
| Spinach               | 282    | 49    |
| Shiitake Mushrooms    | 83     | 30    |
| Apples                | 251    | 30    |
| Carrots               | 257    | 18    |
| Basil                 | 88     | 54    |
| Potatoes              | 52     | 27    |
| Green Beans           | 198    | 91    |
| Blueberries           | 203    | 88    |
| Oranges               | 87     | 78    |
| Watermelons           | 265    | 22    |