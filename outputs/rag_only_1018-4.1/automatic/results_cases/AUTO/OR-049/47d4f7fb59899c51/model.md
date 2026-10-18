Let:
- S = {1, 2, ..., 10} be the set of shelves, indexed by i (ShelfID as in capacity.csv, in order).
- P = {1, 2, ..., 20} be the set of products, indexed by j (order as in products.csv).
- Let product j correspond to the j-th row in products.csv, with ProductName, Value v_j, and Weight w_j.
- Let shelf i correspond to the i-th row in capacity.csv, with ShelfID i and Capacity C_i.
- Decision variables: x_{i,j} = number of units of product j placed on shelf i (integer, x_{i,j} ≥ 0).

Data (in order):

Shelves (capacity.csv):
| i (ShelfID) | C_i (Capacity) |
|-------------|---------------|
| 1           | 5.0           |
| 2           | 7.0           |
| 3           | 6.0           |
| 4           | 8.0           |
| 5           | 5.5           |
| 6           | 9.0           |
| 7           | 6.5           |
| 8           | 7.5           |
| 9           | 8.2           |
| 10          | 5.7           |

Products (products.csv, in order):
| j | ProductName           | v_j (Value) | w_j (Weight) |
|---|-----------------------|-------------|--------------|
| 1 | Smartphone            | 200         | 1.0          |
| 2 | Laptop                | 1500        | 5.0          |
| 3 | Headphones            | 100         | 0.5          |
| 4 | Camera                | 800         | 2.0          |
| 5 | Smartwatch            | 250         | 0.3          |
| 6 | Tablet                | 600         | 1.5          |
| 7 | Bluetooth Speaker     | 150         | 1.0          |
| 8 | Keyboard              | 80          | 0.8          |
| 9 | Mouse                 | 50          | 0.2          |
|10 | Monitor               | 300         | 3.0          |
|11 | Printer               | 400         | 4.0          |
|12 | External Hard Drive   | 120         | 0.5          |
|13 | Router                | 60          | 0.3          |
|14 | Power Bank            | 40          | 0.4          |
|15 | Memory Card           | 30          | 0.05         |
|16 | USB Flash Drive       | 25          | 0.02         |
|17 | Smart Home Hub        | 100         | 0.6          |
|18 | Gaming Console        | 500         | 4.0          |
|19 | Fitness Tracker       | 90          | 0.2          |
|20 | E-Reader              | 180         | 0.5          |

Model:

Variables:
For each i ∈ {1,...,10}, j ∈ {1,...,20}:
 x_{i,j} ∈ {0, 1, 2, ...}

Objective:
Maximize total value across all shelves:
\[
\text{Maximize} \quad Z = \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \cdot x_{i,j}
\]
where v_j is as above.

Constraints:
For each shelf i ∈ {1,...,10} (with capacity C_i):
\[
\sum_{j=1}^{20} w_j \cdot x_{i,j} \leq C_i
\]
where w_j is as above.

Variable domains:
\[
x_{i,j} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,...,10\},\ j \in \{1,...,20\}
\]

Explicitly, for each shelf (i = 1 to 10), the constraint is:
\[
\sum_{j=1}^{20} w_j \cdot x_{i,j} \leq C_i
\]
with the weights and capacities as listed above.

All variables, coefficients, and constraints are directly aligned with the supplied data and the user’s requirements. No additional constraints or assumptions are introduced.

This is a complete integer programming formulation for the described retail shelf allocation problem.