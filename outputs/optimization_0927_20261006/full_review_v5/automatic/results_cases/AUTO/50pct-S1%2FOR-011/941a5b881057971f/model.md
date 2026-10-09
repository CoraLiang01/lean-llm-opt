#### Sets and Indices

Let $i$ index the products in the order given by products.csv:

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

#### Parameters

For each product $i$:

| ProductName           | Value ($v_i$) | Weight ($w_i$) |
|----------------------|:-------------:|:--------------:|
| Spinach              | 64            | 230            |
| Shiitake Mushrooms   | 75            | 637            |
| Apples               | 68            | 773            |
| Carrots              | 11            | 653            |
| Basil                | 91            | 755            |
| Potatoes             | 31            | 670            |
| Green Beans          | 90            | 505            |
| Blueberries          | 56            | 821            |
| Oranges              | 10            | 83             |
| Watermelons          | 24            | 249            |

Total stock capacity: $C = 875$

#### Decision Variables

For each product $i$:

$x_i$ = number of units of product $i$ to order each day

$x_i \in \mathbb{Z}_{\geq 0}$ (nonnegative integers)

#### Mathematical Model

**Objective:**

$$
\max \sum_{i=1}^{10} v_i x_i
$$

That is,

$$
\max \big(
64x_1 + 75x_2 + 68x_3 + 11x_4 + 91x_5 + 31x_6 + 90x_7 + 56x_8 + 10x_9 + 24x_{10}
\big)
$$

**Subject to:**

_Total stock capacity constraint:_

$$
230x_1 + 637x_2 + 773x_3 + 653x_4 + 755x_5 + 670x_6 + 505x_7 + 821x_8 + 83x_9 + 249x_{10} \leq 875
$$

_Nonnegativity and integrality:_

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1, \ldots, 10
$$

#### Parameter Table (from products.csv and capacity.csv, in source order)

| ProductName           | Weight | Value | archive_revision_number | record_keeper_group |
|----------------------|--------|-------|------------------------|--------------------|
| Spinach              | 230    | 64    | 1                      | Team A             |
| Shiitake Mushrooms   | 637    | 75    | 1                      | Team B             |
| Apples               | 773    | 68    | 1                      | Team C             |
| Carrots              | 653    | 11    | 8                      | Team C             |
| Basil                | 755    | 91    | 5                      | Team A             |
| Potatoes             | 670    | 31    | 3                      | Team B             |
| Green Beans          | 505    | 90    | 3                      | Team C             |
| Blueberries          | 821    | 56    | 4                      | Team B             |
| Oranges              | 83     | 10    | 5                      | Team B             |
| Watermelons          | 249    | 24    | 8                      | Team B             |

| archive_revision_number | Capacity |
|------------------------|----------|
| 1                      | 875      |

#### Complete Model

$$
\begin{align*}
\max\ & 64x_1 + 75x_2 + 68x_3 + 11x_4 + 91x_5 + 31x_6 + 90x_7 + 56x_8 + 10x_9 + 24x_{10} \\
\text{s.t.}\quad
& 230x_1 + 637x_2 + 773x_3 + 653x_4 + 755x_5 + 670x_6 + 505x_7 + 821x_8 + 83x_9 + 249x_{10} \leq 875 \\
& x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1, \ldots, 10
\end{align*}
$$