Let $x$ be the number of units of the product with id_number id999 to fulfill.

Objective:
$$
\max\ 434.74\, x
$$

Subject to:
$$
0 \leq x \leq 8171 \\
0 \leq x \leq 56450 \\
x \in \mathbb{Z}_{\geq 0}
$$

Where:
- $x$ = number of units of product id999 fulfilled (integer, nonnegative)
- 434.74 = revenue per unit
- 8171 = demand during the sales horizon
- 56450 = initial inventory available

The fulfillment quantity $x$ cannot exceed either demand or initial inventory.