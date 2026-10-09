##### Decision Variables

- $x_{ij} \geq 0$: quantity supplied from supplier $i \in I$ to supermarket $j \in J$ (continuous).
- $y_i \in \{0,1\}$: whether supplier $i$ is activated (binary).

##### Parameters

- Suppliers: $I = \{\text{S1}, \text{S2}\}$
- Supermarkets: $J = \{\text{C1}, \text{C2}\}$

- Demands:
  - $d_{\text{C1}} = 144$
  - $d_{\text{C2}} = 216$

- Fixed costs:
  - $f_{\text{S1}} = 105.97$
  - $f_{\text{S2}} = 85.31$

- Transportation costs per unit:
  - $c_{\text{S1},\text{C1}} = 2358.39$
  - $c_{\text{S1},\text{C2}} = 1492.08$
  - $c_{\text{S2},\text{C1}} = 0.07$
  - $c_{\text{S2},\text{C2}} = 52.32$

##### Objective Function

\[
\min \left(
    2358.39\,x_{\text{S1},\text{C1}} + 1492.08\,x_{\text{S1},\text{C2}}
  + 0.07\,x_{\text{S2},\text{C1}} + 52.32\,x_{\text{S2},\text{C2}}
  + 105.97\,y_{\text{S1}} + 85.31\,y_{\text{S2}}
\right)
\]

##### Constraints

1. **Demand satisfaction (each supermarket receives exactly its demand):**
   - $x_{\text{S1},\text{C1}} + x_{\text{S2},\text{C1}} = 144$
   - $x_{\text{S1},\text{C2}} + x_{\text{S2},\text{C2}} = 216$

2. **Supplier activation (suppliers can only ship if activated):**
   - $x_{\text{S1},\text{C1}} + x_{\text{S1},\text{C2}} \leq 360\,y_{\text{S1}}$
   - $x_{\text{S2},\text{C1}} + x_{\text{S2},\text{C2}} \leq 360\,y_{\text{S2}}$

   where $360 = 144 + 216$ is the total demand and serves as a valid upper bound.

3. **Variable domains:**
   - $x_{ij} \geq 0$ for all $i \in I$, $j \in J$
   - $y_i \in \{0,1\}$ for all $i \in I$

##### Parameter Tables

**Demands:**

| Customer | Demand |
|----------|--------|
| C1       | 144    |
| C2       | 216    |

**Fixed Costs:**

| Supplier | Fixed Cost |
|----------|------------|
| S1       | 105.97     |
| S2       | 85.31      |

**Transportation Costs:**

| Supplier | C1      | C2     |
|----------|---------|--------|
| S1       | 2358.39 | 1492.08|
| S2       | 0.07    | 52.32  |

**Sets:**

- $I = \{\text{S1}, \text{S2}\}$
- $J = \{\text{C1}, \text{C2}\}$

**Big-M constant:**

- $M = 360$

---

This model determines which suppliers to activate and how much each should supply to each supermarket, minimizing total fixed and transportation costs while meeting all supermarket demands.