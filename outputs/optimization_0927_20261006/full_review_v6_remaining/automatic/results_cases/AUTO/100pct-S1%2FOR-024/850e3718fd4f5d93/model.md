##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from warehouse $i \in I$ to musician/band $j \in J$ (continuous).
- $y_i \in \{0,1\}$: whether warehouse $i$ is activated (binary).

##### Parameters

- $I = \{\text{S1}, \text{S2}, \text{S3}\}$: set of warehouses.
- $J = \{\text{C1}, \text{C2}, \text{C3}\}$: set of musicians/bands.
- Demands:
  - $d_{\text{C1}} = 1083$
  - $d_{\text{C2}} = 776$
  - $d_{\text{C3}} = 16214$
- Fixed costs:
  - $f_{\text{S1}} = 102.33$
  - $f_{\text{S2}} = 94.92$
  - $f_{\text{S3}} = 91.83$
- Transportation costs per unit:
  - $c_{\text{S1},\text{C1}} = 1506.22$, $c_{\text{S1},\text{C2}} = 70.9$, $c_{\text{S1},\text{C3}} = 8.44$
  - $c_{\text{S2},\text{C1}} = 1732.65$, $c_{\text{S2},\text{C2}} = 1780.72$, $c_{\text{S2},\text{C3}} = 567.44$
  - $c_{\text{S3},\text{C1}} = 115.66$, $c_{\text{S3},\text{C2}} = 100.76$, $c_{\text{S3},\text{C3}} = 64.68$
- $M = \sum_{j \in J} d_j = 1083 + 776 + 16214 = 18073$

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:**  
   For each musician/band $j \in J$,
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]
2. **Warehouse activation:**  
   For each warehouse $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq M y_i
   \]
   (where $M = 18073$)
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### Retrieved Information

{
  "warehouses": [
    "S1",
    "S2",
    "S3"
  ],
  "musicians_bands": [
    "C1",
    "C2",
    "C3"
  ],
  "demand": {
    "C1": 1083,
    "C2": 776,
    "C3": 16214
  },
  "fixed_cost": {
    "S1": 102.33,
    "S2": 94.92,
    "S3": 91.83
  },
  "cost": {
    "S1": {
      "C1": 1506.22,
      "C2": 70.9,
      "C3": 8.44
    },
    "S2": {
      "C1": 1732.65,
      "C2": 1780.72,
      "C3": 567.44
    },
    "S3": {
      "C1": 115.66,
      "C2": 100.76,
      "C3": 64.68
    }
  },
  "M": 18073
}