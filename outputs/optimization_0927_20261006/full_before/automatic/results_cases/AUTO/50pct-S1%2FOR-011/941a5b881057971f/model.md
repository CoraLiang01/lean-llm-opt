Let $x_i$ be the number of units of product $i$ to order each day, where $i$ indexes the products in the order given below. All $x_i$ are nonnegative integers.

Objective:
\[
\max \; 64x_1 + 75x_2 + 68x_3 + 11x_4 + 91x_5 + 31x_6 + 90x_7 + 56x_8 + 10x_9 + 24x_{10}
\]

Subject to:

\[
230x_1 + 637x_2 + 773x_3 + 653x_4 + 755x_5 + 670x_6 + 505x_7 + 821x_8 + 83x_9 + 249x_{10} \leq 875
\]

\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1, \ldots, 10
\]

Where the products are indexed as follows (in source order):

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

Parameter values:

- ProductName: Spinach, Value: 64, Weight: 230
- ProductName: Shiitake Mushrooms, Value: 75, Weight: 637
- ProductName: Apples, Value: 68, Weight: 773
- ProductName: Carrots, Value: 11, Weight: 653
- ProductName: Basil, Value: 91, Weight: 755
- ProductName: Potatoes, Value: 31, Weight: 670
- ProductName: Green Beans, Value: 90, Weight: 505
- ProductName: Blueberries, Value: 56, Weight: 821
- ProductName: Oranges, Value: 10, Weight: 83
- ProductName: Watermelons, Value: 24, Weight: 249

Total stock capacity: 875

All variables $x_i$ are nonnegative integers.