Let:
- \( x_{ij} \): number of units of product \( j \) (ProductName as below) placed on display \( i \) (ShelfID as below), for \( i \in \{1,2,\ldots,10\} \), \( j \in \{1,2,\ldots,20\} \).
- All \( x_{ij} \) are nonnegative integers.

Data (in source order):

Displays (from capacity.csv):

| ShelfID | Capacity |
|---------|----------|
| 1       | 5.0      |
| 2       | 7.0      |
| 3       | 6.0      |
| 4       | 8.0      |
| 5       | 5.5      |
| 6       | 9.0      |
| 7       | 6.5      |
| 8       | 7.5      |
| 9       | 8.2      |
| 10      | 5.7      |

Products (from products.csv, in order):

| j | ProductName           | Value | Weight |
|---|-----------------------|-------|--------|
| 1 | Smartphone            | 200   | 1.0    |
| 2 | Laptop                | 1500  | 5.0    |
| 3 | Headphones            | 100   | 0.5    |
| 4 | Camera                | 800   | 2.0    |
| 5 | Smartwatch            | 250   | 0.3    |
| 6 | Tablet                | 600   | 1.5    |
| 7 | Bluetooth Speaker     | 150   | 1.0    |
| 8 | Keyboard              | 80    | 0.8    |
| 9 | Mouse                 | 50    | 0.2    |
|10 | Monitor               | 300   | 3.0    |
|11 | Printer               | 400   | 4.0    |
|12 | External Hard Drive   | 120   | 0.5    |
|13 | Router                | 60    | 0.3    |
|14 | Power Bank            | 40    | 0.4    |
|15 | Memory Card           | 30    | 0.05   |
|16 | USB Flash Drive       | 25    | 0.02   |
|17 | Smart Home Hub        | 100   | 0.6    |
|18 | Gaming Console        | 500   | 4.0    |
|19 | Fitness Tracker       | 90    | 0.2    |
|20 | E-Reader              | 180   | 0.5    |

Model:

Variables:
- \( x_{ij} \in \mathbb{Z}_+ \) (nonnegative integers), for \( i = 1,\ldots,10 \), \( j = 1,\ldots,20 \).

Objective:
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{20} \text{Value}_j \cdot x_{ij}
\]
where \(\text{Value}_j\) is as in the table above for each product \(j\).

Constraints:

1. Display capacity constraints (for each display \(i\)):
\[
\sum_{j=1}^{20} \text{Weight}_j \cdot x_{ij} \leq \text{Capacity}_i \qquad \forall i \in \{1,\ldots,10\}
\]
where \(\text{Weight}_j\) and \(\text{Capacity}_i\) are as in the tables above.

Explicitly, for each shelf:

- Shelf 1: \(\sum_{j=1}^{20} \text{Weight}_j \cdot x_{1j} \leq 5.0\)
- Shelf 2: \(\sum_{j=1}^{20} \text{Weight}_j \cdot x_{2j} \leq 7.0\)
- Shelf 3: \(\sum_{j=1}^{20} \text{Weight}_j \cdot x_{3j} \leq 6.0\)
- Shelf 4: \(\sum_{j=1}^{20} \text{Weight}_j \cdot x_{4j} \leq 8.0\)
- Shelf 5: \(\sum_{j=1}^{20} \text{Weight}_j \cdot x_{5j} \leq 5.5\)
- Shelf 6: \(\sum_{j=1}^{20} \text{Weight}_j \cdot x_{6j} \leq 9.0\)
- Shelf 7: \(\sum_{j=1}^{20} \text{Weight}_j \cdot x_{7j} \leq 6.5\)
- Shelf 8: \(\sum_{j=1}^{20} \text{Weight}_j \cdot x_{8j} \leq 7.5\)
- Shelf 9: \(\sum_{j=1}^{20} \text{Weight}_j \cdot x_{9j} \leq 8.2\)
- Shelf 10: \(\sum_{j=1}^{20} \text{Weight}_j \cdot x_{10j} \leq 5.7\)

2. Minimum total quantity of the first product (Smartphone) across all displays:
\[
\sum_{i=1}^{10} x_{i1} \geq 5
\]

3. Nonnegativity and integrality:
\[
x_{ij} \in \mathbb{Z}_+, \quad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
\]

Summary:
- Decision variables: \( x_{ij} \) = number of units of product \(j\) on display \(i\), integer, \(\geq 0\).
- Objective: maximize total value placed.
- Each display's total weight cannot exceed its capacity.
- At least 5 units of Smartphones (product 1) must be placed in total.

All coefficients and indices are as above, preserving the original file and row order.