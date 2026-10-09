Let:
- S = {1, 2, ..., 10} be the set of ShelfIDs (from capacity.csv)
- P = {Smartphone, Laptop, Headphones, Camera, Smartwatch, Tablet, Bluetooth Speaker, Keyboard, Mouse, Monitor, Printer, External Hard Drive, Router, Power Bank, Memory Card, USB Flash Drive, Smart Home Hub, Gaming Console, Fitness Tracker, E-Reader} be the set of ProductNames (from products.csv)

Decision variables:
For each shelf i ∈ S and product j ∈ P,
 x_{i,j} = number of units of product j placed on shelf i
 x_{i,j} ∈ {0, 1, 2, ...} (nonnegative integers)

Parameters:
From capacity.csv:
 Capacity_i = capacity of shelf i

From products.csv:
 Value_j = value of product j
 Weight_j = weight of product j

Data:
capacity.csv

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

products.csv

| ProductName           | Value | Weight |
|---------------------- |-------|--------|
| Smartphone            | 200   | 1.0    |
| Laptop                | 1500  | 5.0    |
| Headphones            | 100   | 0.5    |
| Camera                | 800   | 2.0    |
| Smartwatch            | 250   | 0.3    |
| Tablet                | 600   | 1.5    |
| Bluetooth Speaker     | 150   | 1.0    |
| Keyboard              | 80    | 0.8    |
| Mouse                 | 50    | 0.2    |
| Monitor               | 300   | 3.0    |
| Printer               | 400   | 4.0    |
| External Hard Drive   | 120   | 0.5    |
| Router                | 60    | 0.3    |
| Power Bank            | 40    | 0.4    |
| Memory Card           | 30    | 0.05   |
| USB Flash Drive       | 25    | 0.02   |
| Smart Home Hub        | 100   | 0.6    |
| Gaming Console        | 500   | 4.0    |
| Fitness Tracker       | 90    | 0.2    |
| E-Reader              | 180   | 0.5    |

Mathematical Model:

Maximize total value across all shelves:
\[
\text{Maximize} \quad Z = \sum_{i \in S} \sum_{j \in P} \text{Value}_j \cdot x_{i,j}
\]

Subject to shelf capacity constraints (for each shelf i):
\[
\sum_{j \in P} \text{Weight}_j \cdot x_{i,j} \leq \text{Capacity}_i \quad \forall i \in S
\]

Variable domains:
\[
x_{i,j} \in \mathbb{Z}_{\geq 0} \quad \forall i \in S, \forall j \in P
\]

Where all coefficients and indices are as given in the tables above.