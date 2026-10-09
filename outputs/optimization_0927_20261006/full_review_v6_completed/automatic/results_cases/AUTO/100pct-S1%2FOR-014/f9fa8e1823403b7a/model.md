Let $x_{ij}$ be the number of units of product $j$ (ProductName from products.csv) to be placed on shelf $i$ (ShelfID from capacity.csv). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- Let $S$ be the set of shelves, indexed by ShelfID:
  $$
  S = \{1, 2, 3, 4, 5, 6, 7, 8, 9, 10\}
  $$
  with capacities:
  \begin{align*}
  \text{Capacity}_1 &= 5.0 \\
  \text{Capacity}_2 &= 7.0 \\
  \text{Capacity}_3 &= 6.0 \\
  \text{Capacity}_4 &= 8.0 \\
  \text{Capacity}_5 &= 5.5 \\
  \text{Capacity}_6 &= 9.0 \\
  \text{Capacity}_7 &= 6.5 \\
  \text{Capacity}_8 &= 7.5 \\
  \text{Capacity}_9 &= 8.2 \\
  \text{Capacity}_{10} &= 5.7 \\
  \end{align*}

- Let $P$ be the set of products, indexed by ProductName:
  \begin{align*}
  &\text{Smartphone:} \quad \text{Value}=200, \quad \text{Weight}=1.0 \\
  &\text{Laptop:} \quad \text{Value}=1500, \quad \text{Weight}=5.0 \\
  &\text{Headphones:} \quad \text{Value}=100, \quad \text{Weight}=0.5 \\
  &\text{Camera:} \quad \text{Value}=800, \quad \text{Weight}=2.0 \\
  &\text{Smartwatch:} \quad \text{Value}=250, \quad \text{Weight}=0.3 \\
  &\text{Tablet:} \quad \text{Value}=600, \quad \text{Weight}=1.5 \\
  &\text{Bluetooth Speaker:} \quad \text{Value}=150, \quad \text{Weight}=1.0 \\
  &\text{Keyboard:} \quad \text{Value}=80, \quad \text{Weight}=0.8 \\
  &\text{Mouse:} \quad \text{Value}=50, \quad \text{Weight}=0.2 \\
  &\text{Monitor:} \quad \text{Value}=300, \quad \text{Weight}=3.0 \\
  &\text{Printer:} \quad \text{Value}=400, \quad \text{Weight}=4.0 \\
  &\text{External Hard Drive:} \quad \text{Value}=120, \quad \text{Weight}=0.5 \\
  &\text{Router:} \quad \text{Value}=60, \quad \text{Weight}=0.3 \\
  &\text{Power Bank:} \quad \text{Value}=40, \quad \text{Weight}=0.4 \\
  &\text{Memory Card:} \quad \text{Value}=30, \quad \text{Weight}=0.05 \\
  &\text{USB Flash Drive:} \quad \text{Value}=25, \quad \text{Weight}=0.02 \\
  &\text{Smart Home Hub:} \quad \text{Value}=100, \quad \text{Weight}=0.6 \\
  &\text{Gaming Console:} \quad \text{Value}=500, \quad \text{Weight}=4.0 \\
  &\text{Fitness Tracker:} \quad \text{Value}=90, \quad \text{Weight}=0.2 \\
  &\text{E-Reader:} \quad \text{Value}=180, \quad \text{Weight}=0.5 \\
  \end{align*}

**Decision Variables:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in S, \forall j \in P
$$

**Objective:**

Maximize the total value of products allocated to all shelves:
$$
\max \sum_{i \in S} \sum_{j \in P} \text{Value}_j \cdot x_{ij}
$$

**Constraints:**

For each shelf $i \in S$, the total weight of products allocated to shelf $i$ cannot exceed its capacity:
$$
\sum_{j \in P} \text{Weight}_j \cdot x_{ij} \leq \text{Capacity}_i \qquad \forall i \in S
$$

**Variable Domains:**

$$
x_{ij} \in \{0, 1, 2, \ldots\} \qquad \forall i \in S, \forall j \in P
$$

**Explicit Data Used:**

- Shelves (ShelfID) and their capacities (Capacity) from capacity.csv, in source order:

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

- Products (ProductName), their values (Value), and weights (Weight) from products.csv, in source order:

  | ProductName           | Value | Weight |
  |-----------------------|-------|--------|
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

This completes the mathematical optimization model as requested.