Let:
- $I = \{P1, P2, \ldots, P111\}$ be the set of products (from unit_product_profits.csv, device_time.csv).
- $K = \{A, B, C, D, E, F, G, H, I, J\}$ be the set of devices (from device_time.csv, monthly_device_capacity.csv).
- $x_i \geq 0$ be the production quantity of product $i \in I$ (continuous, as specified).

Parameters:
- $p_i$: Unit profit of product $i$ (from unit_product_profits.csv).
- $a_{ki}$: Processing time required by product $i$ on device $k$ (from device_time.csv).
- $c_k$: Monthly capacity of device $k$ (from monthly_device_capacity.csv).

---

### Mathematical Model

**Decision Variables:**
- $x_i \geq 0$ (continuous), $\forall i \in I$

**Objective:**
\[
\max \sum_{i \in I} p_i x_i
\]

**Subject to:**

For each device $k \in K$:
\[
\sum_{i \in I} a_{ki} x_i \leq c_k
\]

For all products $i \in I$:
\[
x_i \geq 0
\]

---

### Explicit Formulation with Retrieved Data

Let $I = \{$P1, P2, ..., P111$\}$ and $K = \{$A, B, C, D, E, F, G, H, I, J$\}$.

#### Parameters

- $p_i$ (unit_product_profits.csv):

| Product | Unit_Profit |
|---------|-------------|
| P1      | 28.55       |
| P2      | 12.78       |
| ...     | ...         |
| P111    | 9.99        |

- $a_{ki}$ (device_time.csv): For each device $k$ and product $i$, $a_{ki}$ is the processing time coefficient from the corresponding row and column.

- $c_k$ (monthly_device_capacity.csv):

| Device | Monthly_Capacity |
|--------|------------------|
| A      | 3500             |
| B      | 4200             |
| C      | 4500             |
| D      | 2800             |
| E      | 3300             |
| F      | 3800             |
| G      | 4100             |
| H      | 3900             |
| I      | 4800             |
| J      | 3100             |

#### Model

\[
\begin{align*}
\max \quad & \sum_{i \in I} p_i x_i \\
\text{s.t.} \quad & \sum_{i \in I} a_{ki} x_i \leq c_k, \quad \forall k \in K \\
& x_i \geq 0, \quad \forall i \in I
\end{align*}
\]

Where:
- $p_i$ is the Unit_Profit for product $i$ from unit_product_profits.csv.
- $a_{ki}$ is the processing time for product $i$ on device $k$ from device_time.csv.
- $c_k$ is the Monthly_Capacity for device $k$ from monthly_device_capacity.csv.

**All indices, coefficients, and constraints are as retrieved and aligned by explicit product and device IDs.**