##### Objective Function:

$\quad \max \sum_{i=1}^8 r_i x_i$

where $x_i$ is the number of units of product $i$ to fulfill, and $r_i$ is the revenue per unit for product $i$.

##### Constraints:

For each product $i$ (indexed in the order below):

1. Inventory constraint:
$\quad x_i \leq \text{Initial Inventory}_i$

2. Demand constraint:
$\quad x_i \leq \text{Demand}_i$

3. Non-negativity and integrality:
$\quad x_i \geq 0,\quad x_i \in \mathbb{Z}$

##### Retrieved Information

Let the products and their parameters be:

| $i$ | Product Reference         | $r_i$ (Revenue) | Demand | Initial Inventory |
|-----|--------------------------|-----------------|--------|------------------|
| 1   | ELE-SMA-10000463         | 4.0             | 295    | 2000.0           |
| 2   | ELE-SMA-10000487         | 14.0            | 1002   | 7000.0           |
| 3   | ELE-SMA-10003333         | 14.0            | 958    | 7000.0           |
| 4   | ELE-SMA-10009012         | 4.0             | 777    | 6000.0           |
| 5   | ELE-SMA-10009999         | 4.0             | 271    | 2000.0           |
| 6   | ELE-SMA-10011234         | 4.0             | 244    | 2000.0           |
| 7   | ELE-SMA-10027456         | 14.0            | 990    | 7000.0           |
| 8   | ELE-SMA-10028567         | 14.0            | 1000   | 7000.0           |

##### Full Model

$\max \Big[ 4.0\,x_1 + 14.0\,x_2 + 14.0\,x_3 + 4.0\,x_4 + 4.0\,x_5 + 4.0\,x_6 + 14.0\,x_7 + 14.0\,x_8 \Big]$

subject to:

$\quad 0 \leq x_1 \leq \min(2000.0, 295)$

$\quad 0 \leq x_2 \leq \min(7000.0, 1002)$

$\quad 0 \leq x_3 \leq \min(7000.0, 958)$

$\quad 0 \leq x_4 \leq \min(6000.0, 777)$

$\quad 0 \leq x_5 \leq \min(2000.0, 271)$

$\quad 0 \leq x_6 \leq \min(2000.0, 244)$

$\quad 0 \leq x_7 \leq \min(7000.0, 990)$

$\quad 0 \leq x_8 \leq \min(7000.0, 1000)$

$\quad x_i \in \mathbb{Z},\quad \forall i=1,\ldots,8$

##### Parameter Vectors

- Product References: [ELE-SMA-10000463, ELE-SMA-10000487, ELE-SMA-10003333, ELE-SMA-10009012, ELE-SMA-10009999, ELE-SMA-10011234, ELE-SMA-10027456, ELE-SMA-10028567]
- Revenue: [4.0, 14.0, 14.0, 4.0, 4.0, 4.0, 14.0, 14.0]
- Demand: [295, 1002, 958, 777, 271, 244, 990, 1000]
- Initial Inventory: [2000.0, 7000.0, 7000.0, 6000.0, 2000.0, 2000.0, 7000.0, 7000.0]