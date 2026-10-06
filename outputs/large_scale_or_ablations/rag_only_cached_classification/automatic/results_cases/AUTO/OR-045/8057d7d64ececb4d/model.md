Let the decision variable \( x_i \) denote the number of units of produce type \( i \) to order daily, where \( x_i \) is a nonnegative integer for each produce type \( i \). The produce types and their associated data are as follows (in original file order):

| Index | ProductName          | Weight (per unit) | Value (per unit) |
|-------|----------------------|-------------------|------------------|
| 1     | Spinach              | 282               | 49               |
| 2     | Shiitake Mushrooms   | 83                | 30               |
| 3     | Apples               | 251               | 30               |
| 4     | Carrots              | 257               | 18               |
| 5     | Basil                | 88                | 54               |
| 6     | Potatoes             | 52                | 27               |
| 7     | Green Beans          | 198               | 91               |
| 8     | Blueberries          | 203               | 88               |
| 9     | Oranges              | 87                | 78               |
| 10    | Watermelons          | 265               | 22               |

The overall inventory capacity is 1035 (from capacity.csv).

Mathematical Optimization Model:

Variables:
- For \( i = 1, \ldots, 10 \), let \( x_i \in \mathbb{Z}_{\geq 0} \) (nonnegative integers), representing the number of units of produce type \( i \) to order daily.

Objective:
\[
\text{Maximize} \quad 49x_1 + 30x_2 + 30x_3 + 18x_4 + 54x_5 + 27x_6 + 91x_7 + 88x_8 + 78x_9 + 22x_{10}
\]

Subject to (capacity constraint):
\[
282x_1 + 83x_2 + 251x_3 + 257x_4 + 88x_5 + 52x_6 + 198x_7 + 203x_8 + 87x_9 + 265x_{10} \leq 1035
\]

Variable domains:
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \text{for } i = 1, \ldots, 10
\]

Where the mapping of \( x_i \) to produce type is as per the table above, in the original row order from products.csv.