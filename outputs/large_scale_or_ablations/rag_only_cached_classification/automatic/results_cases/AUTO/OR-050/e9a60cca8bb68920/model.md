Let:
- S = {1, 2, ..., 10} be the set of displays (ShelfID from capacity.csv, in order).
- P = {1, 2, ..., 20} be the set of products (in the order from products.csv).
- Let product 1 be "Smartphone", product 2 be "Laptop", ..., product 20 be "E-Reader".
- Let x_{ij} = number of units of product j placed on display i, for i ∈ S, j ∈ P. These are nonnegative integers.

Parameters:
- Capacity of display i (from capacity.csv):

| i  | ShelfID | Capacity |
|----|---------|----------|
| 1  | 1       | 5.0      |
| 2  | 2       | 7.0      |
| 3  | 3       | 6.0      |
| 4  | 4       | 8.0      |
| 5  | 5       | 5.5      |
| 6  | 6       | 9.0      |
| 7  | 7       | 6.5      |
| 8  | 8       | 7.5      |
| 9  | 9       | 8.2      |
| 10 | 10      | 5.7      |

- Product values and weights (from products.csv):

| j  | ProductName           | Value | Weight |
|----|----------------------|-------|--------|
| 1  | Smartphone           | 200   | 1.0    |
| 2  | Laptop               | 1500  | 5.0    |
| 3  | Headphones           | 100   | 0.5    |
| 4  | Camera               | 800   | 2.0    |
| 5  | Smartwatch           | 250   | 0.3    |
| 6  | Tablet               | 600   | 1.5    |
| 7  | Bluetooth Speaker    | 150   | 1.0    |
| 8  | Keyboard             | 80    | 0.8    |
| 9  | Mouse                | 50    | 0.2    |
| 10 | Monitor              | 300   | 3.0    |
| 11 | Printer              | 400   | 4.0    |
| 12 | External Hard Drive  | 120   | 0.5    |
| 13 | Router               | 60    | 0.3    |
| 14 | Power Bank           | 40    | 0.4    |
| 15 | Memory Card          | 30    | 0.05   |
| 16 | USB Flash Drive      | 25    | 0.02   |
| 17 | Smart Home Hub       | 100   | 0.6    |
| 18 | Gaming Console       | 500   | 4.0    |
| 19 | Fitness Tracker      | 90    | 0.2    |
| 20 | E-Reader             | 180   | 0.5    |

Mathematical Model:

Variables:
- x_{ij} ∈ {0, 1, 2, ...} for all i ∈ S, j ∈ P

Objective:
Maximize total value placed:
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{20} \text{Value}_j \cdot x_{ij}
\]
where Value_j is as above.

Constraints:

1. Display capacity constraints (for each display i):
\[
\sum_{j=1}^{20} \text{Weight}_j \cdot x_{ij} \leq \text{Capacity}_i \quad \forall i = 1, ..., 10
\]
where Weight_j and Capacity_i are as above.

2. Minimum total quantity of the first product ("Smartphone") across all displays:
\[
\sum_{i=1}^{10} x_{i1} \geq 5
\]

3. Nonnegativity and integrality:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i = 1, ..., 10; \; j = 1, ..., 20
\]

All coefficients and identifiers are as given in the CSVs and preserved in order. No data has been omitted or invented.