Let $x_i$ be the number of units of product $i$ to order each day, where $i$ indexes the products in the order given below.

Objective:
$$
\max \; 64x_1 + 75x_2 + 68x_3 + 11x_4 + 91x_5 + 31x_6 + 90x_7 + 56x_8 + 10x_9 + 24x_{10}
$$

Subject to:
$$
230x_1 + 637x_2 + 773x_3 + 653x_4 + 755x_5 + 670x_6 + 505x_7 + 821x_8 + 83x_9 + 249x_{10} \leq 875
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1,\ldots,10
$$

Where the products are, in order:
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

All coefficients (Value and Weight) are taken directly from the retrieved data. The total stock capacity is 875. Each $x_i$ is a nonnegative integer.