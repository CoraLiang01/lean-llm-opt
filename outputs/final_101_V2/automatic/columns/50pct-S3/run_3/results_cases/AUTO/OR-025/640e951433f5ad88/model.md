##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to supermarket $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether supplier $i$ is activated.

##### Parameters

- $I = \{\text{S1}, \text{S2}\}$ (Suppliers)
- $J = \{\text{C1}, \text{C2}\}$ (Supermarkets)

- Demands:
  - $d_{\text{C1}} = 144$
  - $d_{\text{C2}} = 216$

- Fixed costs:
  - $f_{\text{S1}} = 105.97$
  - $f_{\text{S2}} = 85.31$

- Transportation costs:
  - $c_{\text{S1},\text{C1}} = 2358.39$
  - $c_{\text{S1},\text{C2}} = 1492.08$
  - $c_{\text{S2},\text{C1}} = 0.0700$
  - $c_{\text{S2},\text{C2}} = 52.32$

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

That is,

\[
\min \left(2358.39\,x_{\text{S1},\text{C1}} + 1492.08\,x_{\text{S1},\text{C2}} + 0.0700\,x_{\text{S2},\text{C1}} + 52.32\,x_{\text{S2},\text{C2}} + 105.97\,y_{\text{S1}} + 85.31\,y_{\text{S2}}\right)
\]

##### Constraints

1. **Demand satisfaction:**  
   Each supermarket must receive exactly its demand.
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
   That is,
   \[
   x_{\text{S1},\text{C1}} + x_{\text{S2},\text{C1}} = 144
   \]
   \[
   x_{\text{S1},\text{C2}} + x_{\text{S2},\text{C2}} = 216
   \]

2. **Supplier activation:**  
   No shipments from inactive suppliers.
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]
   Where $M = \sum_{j \in J} d_j = 144 + 216 = 360$.

   That is,
   \[
   x_{\text{S1},\text{C1}} + x_{\text{S1},\text{C2}} \leq 360\,y_{\text{S1}}
   \]
   \[
   x_{\text{S2},\text{C1}} + x_{\text{S2},\text{C2}} \leq 360\,y_{\text{S2}}
   \]

3. **Variable domains:**
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Retrieved Information

{
  "suppliers": ["S1", "S2"],
  "supermarkets": ["C1", "C2"],
  "demand": {
    "C1": 144,
    "C2": 216
  },
  "fixed_cost": {
    "S1": 105.97,
    "S2": 85.31
  },
  "cost": {
    "S1": {
      "C1": 2358.39,
      "C2": 1492.08
    },
    "S2": {
      "C1": 0.0700,
      "C2": 52.32
    }
  },
  "M": 360
}