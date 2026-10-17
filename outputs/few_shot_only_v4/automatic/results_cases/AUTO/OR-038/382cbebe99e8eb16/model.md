Let $x_i$ be the number of vehicles of type $i$ to order per day, where $i$ indexes the vehicle types as listed in the data.

**Parameters:**

- Let $V$ be the set of vehicle types:
  $$
  V = \{\text{Sedans},\ \text{SUVs},\ \text{Electric Vehicles},\ \text{Hybrid Vehicles},\ \text{Trucks},\ \text{Sports Cars},\ \text{Compact Cars},\ \text{Luxury Sedans},\ \text{Vans},\ \text{Pickup Trucks}\}
  $$
- For each vehicle type $i \in V$:
    - $b_i$ = benefit coefficient (Value from products.csv)
    - $u_i$ = daily inventory limit (Capacity from capacity.csv)

**Decision Variables:**

- $x_i \in \mathbb{Z}_{\geq 0}$, for all $i \in V$ (number of vehicles of type $i$ to order per day)

**Model:**

Maximize total benefit:
$$
\max \sum_{i \in V} b_i x_i
$$

Subject to:

1. **Total inventory capacity constraint:**
   $$
   \sum_{i \in V} x_i \leq \sum_{i \in V} u_i
   $$
   (The sum of all ordered units does not exceed the total inventory capacity, which is the sum of all daily inventory limits.)

2. **Daily inventory limit for each vehicle type:**
   $$
   0 \leq x_i \leq u_i, \quad \forall i \in V
   $$

3. **Integrality:**
   $$
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in V
   $$

**Numerical Data:**

| VehicleType         | $b_i$ (Value) | $u_i$ (Capacity) |
|---------------------|:-------------:|:----------------:|
| Sedans              | 1200          | 100              |
| SUVs                | 1800          | 80               |
| Electric Vehicles   | 2500          | 120              |
| Hybrid Vehicles     | 2000          | 90               |
| Trucks              | 1500          | 50               |
| Sports Cars         | 3000          | 30               |
| Compact Cars        | 1000          | 110              |
| Luxury Sedans       | 3500          | 40               |
| Vans                | 1600          | 60               |
| Pickup Trucks       | 1700          | 35               |

**Complete Model:**

$$
\begin{align*}
\max\quad & 1200\,x_{\text{Sedans}} + 1800\,x_{\text{SUVs}} + 2500\,x_{\text{Electric Vehicles}} + 2000\,x_{\text{Hybrid Vehicles}} \\
& + 1500\,x_{\text{Trucks}} + 3000\,x_{\text{Sports Cars}} + 1000\,x_{\text{Compact Cars}} + 3500\,x_{\text{Luxury Sedans}} \\
& + 1600\,x_{\text{Vans}} + 1700\,x_{\text{Pickup Trucks}} \\
\text{s.t.}\quad & x_{\text{Sedans}} + x_{\text{SUVs}} + x_{\text{Electric Vehicles}} + x_{\text{Hybrid Vehicles}} \\
& + x_{\text{Trucks}} + x_{\text{Sports Cars}} + x_{\text{Compact Cars}} + x_{\text{Luxury Sedans}} \\
& + x_{\text{Vans}} + x_{\text{Pickup Trucks}} \leq 715 \\
& 0 \leq x_{\text{Sedans}} \leq 100 \\
& 0 \leq x_{\text{SUVs}} \leq 80 \\
& 0 \leq x_{\text{Electric Vehicles}} \leq 120 \\
& 0 \leq x_{\text{Hybrid Vehicles}} \leq 90 \\
& 0 \leq x_{\text{Trucks}} \leq 50 \\
& 0 \leq x_{\text{Sports Cars}} \leq 30 \\
& 0 \leq x_{\text{Compact Cars}} \leq 110 \\
& 0 \leq x_{\text{Luxury Sedans}} \leq 40 \\
& 0 \leq x_{\text{Vans}} \leq 60 \\
& 0 \leq x_{\text{Pickup Trucks}} \leq 35 \\
& x_i \in \mathbb{Z}_{\geq 0},\quad \forall i \in V
\end{align*}
$$