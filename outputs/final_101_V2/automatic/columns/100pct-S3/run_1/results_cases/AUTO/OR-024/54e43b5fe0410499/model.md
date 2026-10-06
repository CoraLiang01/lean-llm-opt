##### Decision Variables

$x_{ij} \geq 0$: Quantity of goods shipped from warehouse (facility) $i \in I$ to musician/band (customer) $j \in J$ (continuous).
$y_i \in \{0,1\}$: 1 if warehouse $i$ is activated, 0 otherwise.

##### Parameters

- Facilities (warehouses): $I = \{\text{S1}, \text{S2}, \text{S3}\}$
- Customers (musicians/bands): $J = \{\text{C1}, \text{C2}, \text{C3}\}$

- Demand:
  - $d_{\text{C1}} = 1083$
  - $d_{\text{C2}} = 776$
  - $d_{\text{C3}} = 16214$

- Fixed costs:
  - $f_{\text{S1}} = 102.33$
  - $f_{\text{S2}} = 94.92$
  - $f_{\text{S3}} = 91.83$

- Transportation costs $c_{ij}$ (per unit):
  - $c_{\text{S1},\text{C1}} = 1506.22$
  - $c_{\text{S1},\text{C2}} = 70.9$
  - $c_{\text{S1},\text{C3}} = 8.44$
  - $c_{\text{S2},\text{C1}} = 1732.65$
  - $c_{\text{S2},\text{C2}} = 1780.72$
  - $c_{\text{S2},\text{C3}} = 567.44$
  - $c_{\text{S3},\text{C1}} = 115.66$
  - $c_{\text{S3},\text{C2}} = 100.76$
  - $c_{\text{S3},\text{C3}} = 64.68$

- $M = \sum_{j \in J} d_j = 1083 + 776 + 16214 = 18073$

##### Objective Function

\[
\min \left(
\sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
+ \sum_{i \in I} f_i y_i
\right)
\]

That is,

\[
\min \Big(
1506.22\,x_{\text{S1},\text{C1}} + 70.9\,x_{\text{S1},\text{C2}} + 8.44\,x_{\text{S1},\text{C3}}
+ 1732.65\,x_{\text{S2},\text{C1}} + 1780.72\,x_{\text{S2},\text{C2}} + 567.44\,x_{\text{S2},\text{C3}}
+ 115.66\,x_{\text{S3},\text{C1}} + 100.76\,x_{\text{S3},\text{C2}} + 64.68\,x_{\text{S3},\text{C3}}
+ 102.33\,y_{\text{S1}} + 94.92\,y_{\text{S2}} + 91.83\,y_{\text{S3}}
\Big)
\]

##### Constraints

1. **Demand satisfaction:**  
   For each customer $j \in J$,
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]
   That is,
   - $x_{\text{S1},\text{C1}} + x_{\text{S2},\text{C1}} + x_{\text{S3},\text{C1}} = 1083$
   - $x_{\text{S1},\text{C2}} + x_{\text{S2},\text{C2}} + x_{\text{S3},\text{C2}} = 776$
   - $x_{\text{S1},\text{C3}} + x_{\text{S2},\text{C3}} + x_{\text{S3},\text{C3}} = 16214$

2. **Facility activation:**  
   For each facility $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq M\, y_i
   \]
   That is,
   - $x_{\text{S1},\text{C1}} + x_{\text{S1},\text{C2}} + x_{\text{S1},\text{C3}} \leq 18073\, y_{\text{S1}}$
   - $x_{\text{S2},\text{C1}} + x_{\text{S2},\text{C2}} + x_{\text{S2},\text{C3}} \leq 18073\, y_{\text{S2}}$
   - $x_{\text{S3},\text{C1}} + x_{\text{S3},\text{C2}} + x_{\text{S3},\text{C3}} \leq 18073\, y_{\text{S3}}$

3. **Variable domains:**
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

---

###### Retrieved Information

{
  "facilities": ["S1", "S2", "S3"],
  "customers": ["C1", "C2", "C3"],
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
    "S1": {"C1": 1506.22, "C2": 70.9, "C3": 8.44},
    "S2": {"C1": 1732.65, "C2": 1780.72, "C3": 567.44},
    "S3": {"C1": 115.66, "C2": 100.76, "C3": 64.68}
  },
  "M": 18073
}