Let $x_i$ denote the production quantity (in continuous, nonnegative units) of product $i$ ($i = \text{P1}, \ldots, \text{P111}$).

**Parameters:**

- $p_i$: Unit_Profit of product $i$ (from unit_product_profits.csv)
- $a_{ji}$: Processing time required by product $i$ on device $j$ (from device_time.csv; $j = \text{A}, \ldots, \text{J}$)
- $c_j$: Monthly_Capacity of device $j$ (from monthly_device_capacity.csv)

**Model:**

Maximize total monthly profit:
$$
\max \sum_{i=\text{P1}}^{\text{P111}} p_i x_i
$$

Subject to device capacity constraints (for each device $j$):
$$
\sum_{i=\text{P1}}^{\text{P111}} a_{ji} x_i \leq c_j \qquad \forall j \in \{\text{A}, \text{B}, \text{C}, \text{D}, \text{E}, \text{F}, \text{G}, \text{H}, \text{I}, \text{J}\}
$$

Nonnegativity:
$$
x_i \geq 0 \qquad \forall i \in \{\text{P1}, \ldots, \text{P111}\}
$$

**Parameter values (retrieved):**

- Products: P1, P2, ..., P111
- Devices: A, B, C, D, E, F, G, H, I, J

- $p_i$ (Unit_Profit for each product $i$):

| Product | Unit_Profit |
|---------|-------------|
| P1      | 28.55       |
| P2      | 12.78       |
| P3      | 45.21       |
| ...     | ...         |
| P111    | 9.99        |

- $a_{ji}$ (Processing time required by product $i$ on device $j$):

(See device_time.csv; for example, for device A and product P1, $a_{\text{A},\text{P1}} = 8.1$; for device B and product P2, $a_{\text{B},\text{P2}} = 5.2$, etc.)

- $c_j$ (Monthly_Capacity for each device $j$):

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

**Decision variables:**

$$
x_i \geq 0, \quad x_i \in \mathbb{R} \qquad \forall i \in \{\text{P1}, \ldots, \text{P111}\}
$$

**Complete Model:**

$$
\begin{align*}
\max \quad & \sum_{i=\text{P1}}^{\text{P111}} p_i x_i \\
\text{s.t.} \quad & \sum_{i=\text{P1}}^{\text{P111}} a_{ji} x_i \leq c_j \qquad \forall j \in \{\text{A}, \text{B}, \text{C}, \text{D}, \text{E}, \text{F}, \text{G}, \text{H}, \text{I}, \text{J}\} \\
& x_i \geq 0 \qquad \forall i \in \{\text{P1}, \ldots, \text{P111}\}
\end{align*}
$$

Where all coefficients and identifiers are as given in the retrieved CSV files.