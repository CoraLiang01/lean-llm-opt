Let:
- ShelfID i ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
- Product j ∈ {1: Smartphone, 2: Laptop, 3: Headphones, 4: Camera, 5: Smartwatch, 6: Tablet, 7: Bluetooth Speaker, 8: Keyboard, 9: Mouse, 10: Monitor, 11: Printer, 12: External Hard Drive, 13: Router, 14: Power Bank, 15: Memory Card, 16: USB Flash Drive, 17: Smart Home Hub, 18: Gaming Console, 19: Fitness Tracker, 20: E-Reader}

Parameters:
- Capacity of shelf i: C_i

From capacity.csv:
C_1 = 5.0, C_2 = 7.0, C_3 = 6.0, C_4 = 8.0, C_5 = 5.5, C_6 = 9.0, C_7 = 6.5, C_8 = 7.5, C_9 = 8.2, C_10 = 5.7

- Value of product j: v_j
- Weight of product j: w_j

From products.csv:
| j  | ProductName            | v_j  | w_j   |
|----|------------------------|------|-------|
| 1  | Smartphone             | 200  | 1.0   |
| 2  | Laptop                 | 1500 | 5.0   |
| 3  | Headphones             | 100  | 0.5   |
| 4  | Camera                 | 800  | 2.0   |
| 5  | Smartwatch             | 250  | 0.3   |
| 6  | Tablet                 | 600  | 1.5   |
| 7  | Bluetooth Speaker      | 150  | 1.0   |
| 8  | Keyboard               | 80   | 0.8   |
| 9  | Mouse                  | 50   | 0.2   |
| 10 | Monitor                | 300  | 3.0   |
| 11 | Printer                | 400  | 4.0   |
| 12 | External Hard Drive    | 120  | 0.5   |
| 13 | Router                 | 60   | 0.3   |
| 14 | Power Bank             | 40   | 0.4   |
| 15 | Memory Card            | 30   | 0.05  |
| 16 | USB Flash Drive        | 25   | 0.02  |
| 17 | Smart Home Hub         | 100  | 0.6   |
| 18 | Gaming Console         | 500  | 4.0   |
| 19 | Fitness Tracker        | 90   | 0.2   |
| 20 | E-Reader               | 180  | 0.5   |

Decision variables:
x_{ij} = number of units of product j placed on shelf i, for i ∈ {1,...,10}, j ∈ {1,...,20}
x_{ij} ∈ {0, 1, 2, ...} (nonnegative integers)

Objective:
Maximize total value:
Maximize ∑_{i=1}^{10} ∑_{j=1}^{20} v_j x_{ij}
= ∑_{i=1}^{10} [200 x_{i1} + 1500 x_{i2} + 100 x_{i3} + 800 x_{i4} + 250 x_{i5} + 600 x_{i6} + 150 x_{i7} + 80 x_{i8} + 50 x_{i9} + 300 x_{i10} + 400 x_{i11} + 120 x_{i12} + 60 x_{i13} + 40 x_{i14} + 30 x_{i15} + 25 x_{i16} + 100 x_{i17} + 500 x_{i18} + 90 x_{i19} + 180 x_{i20}]

Subject to:

1. Shelf capacity constraints (for each shelf i):
  ∑_{j=1}^{20} w_j x_{ij} ≤ C_i  for i = 1,...,10

Explicitly:
For i=1: 1.0 x_{1,1} + 5.0 x_{1,2} + 0.5 x_{1,3} + 2.0 x_{1,4} + 0.3 x_{1,5} + 1.5 x_{1,6} + 1.0 x_{1,7} + 0.8 x_{1,8} + 0.2 x_{1,9} + 3.0 x_{1,10} + 4.0 x_{1,11} + 0.5 x_{1,12} + 0.3 x_{1,13} + 0.4 x_{1,14} + 0.05 x_{1,15} + 0.02 x_{1,16} + 0.6 x_{1,17} + 4.0 x_{1,18} + 0.2 x_{1,19} + 0.5 x_{1,20} ≤ 5.0

For i=2: (same as above, with x_{2,j}, ≤ 7.0)
For i=3: (same, ≤ 6.0)
For i=4: (same, ≤ 8.0)
For i=5: (same, ≤ 5.5)
For i=6: (same, ≤ 9.0)
For i=7: (same, ≤ 6.5)
For i=8: (same, ≤ 7.5)
For i=9: (same, ≤ 8.2)
For i=10: (same, ≤ 5.7)

2. Minimum total quantity of the first product (Smartphone) across all shelves:
  ∑_{i=1}^{10} x_{i1} ≥ 5

3. Nonnegativity and integrality:
  x_{ij} ∈ {0, 1, 2, ...}  for all i, j

Summary:
Maximize
  ∑_{i=1}^{10} ∑_{j=1}^{20} v_j x_{ij}
Subject to
  ∑_{j=1}^{20} w_j x_{ij} ≤ C_i  for i = 1,...,10
  ∑_{i=1}^{10} x_{i1} ≥ 5
  x_{ij} ∈ {0, 1, 2, ...}  for all i, j

Where all coefficients and indices are as above, using the explicit values from the CSVs.