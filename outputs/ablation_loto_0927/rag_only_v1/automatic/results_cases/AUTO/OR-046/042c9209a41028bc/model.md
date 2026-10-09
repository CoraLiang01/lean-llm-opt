Let the index i run over the following products in the given order:
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

Let \( x_i \) = number of units of product i to order each day (integer, \( x_i \geq 0 \)).

Parameters (from products.csv and capacity.csv):

| i  | Product Name         | Weight (per unit) | Value (per unit) |
|----|----------------------|-------------------|------------------|
| 1  | Spinach              | 230               | 64               |
| 2  | Shiitake Mushrooms   | 637               | 75               |
| 3  | Apples               | 773               | 68               |
| 4  | Carrots              | 653               | 11               |
| 5  | Basil                | 755               | 91               |
| 6  | Potatoes             | 670               | 31               |
| 7  | Green Beans          | 505               | 90               |
| 8  | Blueberries          | 821               | 56               |
| 9  | Oranges              | 83                | 10               |
| 10 | Watermelons          | 249               | 24               |

Total stock capacity: 875

Mathematical Model:

Decision variables:
\[
x_i \in \mathbb{Z}_+, \quad \text{for } i = 1, \ldots, 10
\]

Objective (maximize total value):
\[
\max \; 64x_1 + 75x_2 + 68x_3 + 11x_4 + 91x_5 + 31x_6 + 90x_7 + 56x_8 + 10x_9 + 24x_{10}
\]

Subject to (total weight does not exceed capacity):
\[
230x_1 + 637x_2 + 773x_3 + 653x_4 + 755x_5 + 670x_6 + 505x_7 + 821x_8 + 83x_9 + 249x_{10} \leq 875
\]

\[
x_i \geq 0, \quad x_i \in \mathbb{Z}, \quad \forall i = 1, \ldots, 10
\]

Where:
- \( x_i \) = number of units of product i to order each day (integer, nonnegative)
- All coefficients and capacity are taken directly from the provided CSV data.