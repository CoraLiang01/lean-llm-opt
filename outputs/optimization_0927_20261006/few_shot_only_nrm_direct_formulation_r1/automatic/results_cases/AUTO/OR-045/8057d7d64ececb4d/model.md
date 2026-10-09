Let $x_i$ be the number of units of produce $i$ to order daily, for $i$ corresponding to the following products:

| $i$ | ProductName           | Weight | Value |
|-----|----------------------|--------|-------|
| 1   | Spinach              | 282    | 49    |
| 2   | Shiitake Mushrooms   | 83     | 30    |
| 3   | Apples               | 251    | 30    |
| 4   | Carrots              | 257    | 18    |
| 5   | Basil                | 88     | 54    |
| 6   | Potatoes             | 52     | 27    |
| 7   | Green Beans          | 198    | 91    |
| 8   | Blueberries          | 203    | 88    |
| 9   | Oranges              | 87     | 78    |
| 10  | Watermelons          | 265    | 22    |

**Objective:**
\[
\max \; 49x_1 + 30x_2 + 30x_3 + 18x_4 + 54x_5 + 27x_6 + 91x_7 + 88x_8 + 78x_9 + 22x_{10}
\]

**Subject to:**
\[
282x_1 + 83x_2 + 251x_3 + 257x_4 + 88x_5 + 52x_6 + 198x_7 + 203x_8 + 87x_9 + 265x_{10} \leq 1035
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \text{for } i = 1,2,\ldots,10
\]