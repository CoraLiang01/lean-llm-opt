Let:
- S = {1, 2, ..., 10} be the set of shelves, indexed by ShelfID as in capacity.csv.
- P = {1, 2, ..., 20} be the set of products, indexed by ProductName as in products.csv.
- x_{ij} = number of units of product j ∈ P placed on shelf i ∈ S (decision variable, integer, x_{ij} ≥ 0).

Parameters (from the data, in original file order):

Shelf capacities (from capacity.csv):
- Shelf 1: Capacity = 750
- Shelf 2: Capacity = 820
- Shelf 3: Capacity = 570
- Shelf 4: Capacity = 800
- Shelf 5: Capacity = 550
- Shelf 6: Capacity = 900
- Shelf 7: Capacity = 650
- Shelf 8: Capacity = 800
- Shelf 9: Capacity = 850
- Shelf 10: Capacity = 900

Product values and weights (from products.csv, in order):
| ProductName (j) | Value v_j | Weight w_j |
|-----------------|-----------|------------|
| 1               | 55        | 10         |
| 2               | 75        | 20         |
| 3               | 65        | 5          |
| 4               | 60        | 15         |
| 5               | 80        | 25         |
| 6               | 90        | 35         |
| 7               | 40        | 45         |
| 8               | 100       | 55         |
| 9               | 55        | 65         |
| 10              | 75        | 20         |
| 11              | 110       | 18         |
| 12              | 50        | 28         |
| 13              | 60        | 8          |
| 14              | 120       | 28         |
| 15              | 70        | 25         |
| 16              | 110       | 40         |
| 17              | 50        | 55         |
| 18              | 60        | 70         |
| 19              | 120       | 85         |
| 20              | 100       | 100        |

Mathematical Model:

Variables:
- For each shelf i ∈ {1,...,10} and product j ∈ {1,...,20}, x_{ij} ∈ {0, 1, 2, ...}

Objective:
Maximize total value of products on all shelves:
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \cdot x_{ij}
\]
where v_j is the Value for product j as above.

Constraints:
For each shelf i ∈ {1,...,10} (using the original order and capacities):

1. Shelf 1 (Capacity 750):
\[
\sum_{j=1}^{20} w_j \cdot x_{1j} \leq 750
\]
2. Shelf 2 (Capacity 820):
\[
\sum_{j=1}^{20} w_j \cdot x_{2j} \leq 820
\]
3. Shelf 3 (Capacity 570):
\[
\sum_{j=1}^{20} w_j \cdot x_{3j} \leq 570
\]
4. Shelf 4 (Capacity 800):
\[
\sum_{j=1}^{20} w_j \cdot x_{4j} \leq 800
\]
5. Shelf 5 (Capacity 550):
\[
\sum_{j=1}^{20} w_j \cdot x_{5j} \leq 550
\]
6. Shelf 6 (Capacity 900):
\[
\sum_{j=1}^{20} w_j \cdot x_{6j} \leq 900
\]
7. Shelf 7 (Capacity 650):
\[
\sum_{j=1}^{20} w_j \cdot x_{7j} \leq 650
\]
8. Shelf 8 (Capacity 800):
\[
\sum_{j=1}^{20} w_j \cdot x_{8j} \leq 800
\]
9. Shelf 9 (Capacity 850):
\[
\sum_{j=1}^{20} w_j \cdot x_{9j} \leq 850
\]
10. Shelf 10 (Capacity 900):
\[
\sum_{j=1}^{20} w_j \cdot x_{10j} \leq 900
\]

where w_j is the Weight for product j as above.

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
\]

Summary:
- Decision variables: x_{ij} = number of units of product j on shelf i (nonnegative integer).
- Objective: Maximize total value placed on all shelves.
- Constraints: For each shelf, total weight of products placed does not exceed its capacity.
- All coefficients and indices are as given in the original data and order.