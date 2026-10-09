Let $x_i$ be the number of units of produce type $i$ to order daily, where $i$ indexes the following products in the order given:

1. Spinach
2. Shiitake Mushrooms
3. Apples
4. Carrots
5. Basil
6. Potatoes
7. Green Beans
8. Blueberries
9. Oranges
10. Watermelons

Let the benefit (value) and weight per unit for each product be as follows:

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

The total inventory capacity is 1035.

The mathematical model is:

Objective:
$$
\max \; 49x_1 + 30x_2 + 30x_3 + 18x_4 + 54x_5 + 27x_6 + 91x_7 + 88x_8 + 78x_9 + 22x_{10}
$$

Subject to:
$$
282x_1 + 83x_2 + 251x_3 + 257x_4 + 88x_5 + 52x_6 + 198x_7 + 203x_8 + 87x_9 + 265x_{10} \leq 1035
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1,2,\ldots,10
$$

Where:
- $x_i$ is the number of units of product $i$ to order daily (integer, nonnegative)
- The coefficients in the objective are the Value for each product
- The coefficients in the constraint are the Weight for each product
- The right-hand side of the constraint is the total inventory Capacity

All data and identifiers are preserved in source order.