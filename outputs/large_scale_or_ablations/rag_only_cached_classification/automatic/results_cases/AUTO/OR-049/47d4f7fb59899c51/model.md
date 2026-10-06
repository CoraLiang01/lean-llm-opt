Let:
- S = {1, 2, ..., 10} be the set of shelves, indexed by i (from capacity.csv, ShelfID).
- P = {1, 2, ..., 20} be the set of products, indexed by j (from products.csv, row order).
- Let product j correspond to the j-th row in products.csv, with ProductName, Value v_j, and Weight w_j.
- Let c_i be the capacity of shelf i from capacity.csv.

Decision variables:
x_ij = number of units of product j placed on shelf i, for all i ∈ S, j ∈ P.
x_ij ∈ {0, 1, 2, ...} (nonnegative integers)

Data:
From capacity.csv:
ShelfID (i) | Capacity (c_i)
1 | 5.0
2 | 7.0
3 | 6.0
4 | 8.0
5 | 5.5
6 | 9.0
7 | 6.5
8 | 7.5
9 | 8.2
10 | 5.7

From products.csv (j = row number, 1-based):
j | ProductName             | Value (v_j) | Weight (w_j)
1 | Smartphone              | 200         | 1.0
2 | Laptop                  | 1500        | 5.0
3 | Headphones              | 100         | 0.5
4 | Camera                  | 800         | 2.0
5 | Smartwatch              | 250         | 0.3
6 | Tablet                  | 600         | 1.5
7 | Bluetooth Speaker       | 150         | 1.0
8 | Keyboard                | 80          | 0.8
9 | Mouse                   | 50          | 0.2
10| Monitor                 | 300         | 3.0
11| Printer                 | 400         | 4.0
12| External Hard Drive     | 120         | 0.5
13| Router                  | 60          | 0.3
14| Power Bank              | 40          | 0.4
15| Memory Card             | 30          | 0.05
16| USB Flash Drive         | 25          | 0.02
17| Smart Home Hub          | 100         | 0.6
18| Gaming Console          | 500         | 4.0
19| Fitness Tracker         | 90          | 0.2
20| E-Reader                | 180         | 0.5

Mathematical Model:

Variables:
x_ij ∈ {0, 1, 2, ...} for i = 1,...,10; j = 1,...,20

Objective:
Maximize total value across all shelves:
\[
\text{Maximize} \quad Z = \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij}
\]
where v_j is as above.

Subject to (for each shelf i):
\[
\sum_{j=1}^{20} w_j x_{ij} \leq c_i \qquad \forall i = 1,...,10
\]
where w_j and c_i are as above.

x_ij ∈ {0, 1, 2, ...} for all i, j.

Explicitly, for each shelf i (using the data):

For shelf 1 (Capacity 5.0):
1.0 x_{1,1} + 5.0 x_{1,2} + 0.5 x_{1,3} + 2.0 x_{1,4} + 0.3 x_{1,5} + 1.5 x_{1,6} + 1.0 x_{1,7} + 0.8 x_{1,8} + 0.2 x_{1,9} + 3.0 x_{1,10} + 4.0 x_{1,11} + 0.5 x_{1,12} + 0.3 x_{1,13} + 0.4 x_{1,14} + 0.05 x_{1,15} + 0.02 x_{1,16} + 0.6 x_{1,17} + 4.0 x_{1,18} + 0.2 x_{1,19} + 0.5 x_{1,20} ≤ 5.0

Repeat similarly for shelves 2 through 10, using their respective capacities.

Summary:
Maximize
\[
Z = \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij}
\]
Subject to, for each i = 1,...,10:
\[
\sum_{j=1}^{20} w_j x_{ij} \leq c_i
\]
x_{ij} ∈ {0, 1, 2, ...} for all i, j.

Where all coefficients and indices are as above, and all data from both CSVs is used in the formulation.