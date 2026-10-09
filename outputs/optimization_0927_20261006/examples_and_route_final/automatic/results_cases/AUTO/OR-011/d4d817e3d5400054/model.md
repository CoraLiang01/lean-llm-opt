Let $x_i$ denote the number of units of product $i$ (with $id\_number = \text{id999}$) to fulfill.

Objective:
$$
\max\ 434.74\, x_i
$$

Subject to:
$$
x_i \leq 56450 \\
x_i \leq 8171 \\
x_i \geq 0,\quad x_i \in \mathbb{Z}
$$

Where:
- $x_i$ = number of units of product id999 fulfilled (integer, $x_i \geq 0$)
- $434.74$ = revenue per unit
- $56450$ = initial inventory
- $8171$ = demand during the sales horizon