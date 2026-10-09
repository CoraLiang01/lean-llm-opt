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

Let $v_i$ be the value (benefit) per unit and $w_i$ the weight per unit for product $i$.

The model is:

**Objective:**
\[
\max \; 49x_1 + 30x_2 + 30x_3 + 18x_4 + 54x_5 + 27x_6 + 91x_7 + 88x_8 + 78x_9 + 22x_{10}
\]

**Subject to:**
\[
282x_1 + 83x_2 + 251x_3 + 257x_4 + 88x_5 + 52x_6 + 198x_7 + 203x_8 + 87x_9 + 265x_{10} \leq 1035
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1, \ldots, 10
\]

**Parameter Table (in source order):**

| ProductName           | $i$ | $w_i$ | $v_i$ |
|-----------------------|-----|-------|-------|
| Spinach               | 1   | 282   | 49    |
| Shiitake Mushrooms    | 2   | 83    | 30    |
| Apples                | 3   | 251   | 30    |
| Carrots               | 4   | 257   | 18    |
| Basil                 | 5   | 88    | 54    |
| Potatoes              | 6   | 52    | 27    |
| Green Beans           | 7   | 198   | 91    |
| Blueberries           | 8   | 203   | 88    |
| Oranges               | 9   | 87    | 78    |
| Watermelons           | 10  | 265   | 22    |

**Capacity:**
\[
\text{Total capacity} = 1035
\]

**Decision variables:**
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \text{for each product } i
\]