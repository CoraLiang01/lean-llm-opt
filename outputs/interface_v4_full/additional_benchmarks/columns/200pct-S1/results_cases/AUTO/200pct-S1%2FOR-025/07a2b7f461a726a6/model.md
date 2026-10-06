##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to supermarket $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether supplier $i$ is activated.

##### Parameters

- Facilities (Suppliers): $I = \{\text{S1}, \text{S2}\}$
- Supermarkets (Customers): $J = \{\text{C1}, \text{C2}\}$
- Demand:
  - $d_{\text{C1}} = 144$
  - $d_{\text{C2}} = 216$
- Fixed Costs:
  - $f_{\text{S1}} = 105.97$
  - $f_{\text{S2}} = 85.31$
- Transportation Costs:
  - $c_{\text{S1},\text{C1}} = 2358.39$
  - $c_{\text{S1},\text{C2}} = 1492.08$
  - $c_{\text{S2},\text{C1}} = 0.07$
  - $c_{\text{S2},\text{C2}} = 52.32$
- $M = d_{\text{C1}} + d_{\text{C2}} = 360$ (sufficiently large upper bound for linking constraints)

##### Objective Function

\[
\min \left[
    2358.39\,x_{\text{S1},\text{C1}} + 1492.08\,x_{\text{S1},\text{C2}}
  + 0.07\,x_{\text{S2},\text{C1}} + 52.32\,x_{\text{S2},\text{C2}}
  + 105.97\,y_{\text{S1}} + 85.31\,y_{\text{S2}}
\right]
\]

##### Constraints

1. **Demand satisfaction (each supermarket receives exactly its demand):**
   - $x_{\text{S1},\text{C1}} + x_{\text{S2},\text{C1}} = 144$
   - $x_{\text{S1},\text{C2}} + x_{\text{S2},\text{C2}} = 216$

2. **Supplier activation (no shipments from inactive suppliers):**
   - $x_{\text{S1},\text{C1}} + x_{\text{S1},\text{C2}} \leq 360\,y_{\text{S1}}$
   - $x_{\text{S2},\text{C1}} + x_{\text{S2},\text{C2}} \leq 360\,y_{\text{S2}}$

3. **Variable domains:**
   - $x_{ij} \geq 0$ for all $i \in I$, $j \in J$
   - $y_i \in \{0,1\}$ for all $i \in I$

##### Full Data Used

- Facilities: S1 (FixedCost: 105.97, Row 3), S2 (FixedCost: 85.31, Row 4)
- Supermarkets: C1 (Demand: 144, Row 1), C2 (Demand: 216, Row 2)
- Transportation Costs:
    - S1→C1: 2358.39 (Row 5)
    - S1→C2: 1492.08 (Row 5)
    - S2→C1: 0.07 (Row 6)
    - S2→C2: 52.32 (Row 6)
- $M = 360$ (sum of all supermarket demands)

All identifiers, coefficients, and matrix orientations are preserved as in the original data. No supplier capacity limits are imposed beyond the linking constraints.