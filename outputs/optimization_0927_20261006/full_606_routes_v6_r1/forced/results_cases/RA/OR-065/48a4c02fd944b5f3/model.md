Let:
- $S_i$ denote warehouse $i$ (with $i \in \{\text{S1}, \text{S2}, \text{S3}\}$)
- $C_j$ denote customer (musician/band) $j$ (with $j \in \{\text{C1}, \text{C2}, \text{C3}\}$)
- $y_i \in \{0,1\}$: 1 if warehouse $S_i$ is operational, 0 otherwise
- $x_{ij} \geq 0$: quantity supplied from warehouse $S_i$ to customer $C_j$

Parameters:
- Fixed costs:
  - $f_{\text{S1}} = 102.33$
  - $f_{\text{S2}} = 94.92$
  - $f_{\text{S3}} = 91.83$
- Demands:
  - $d_{\text{C1}} = 1083$
  - $d_{\text{C2}} = 776$
  - $d_{\text{C3}} = 16214$
- Transportation costs per unit:
  - $c_{\text{S1},\text{C1}} = 1506.22$, $c_{\text{S1},\text{C2}} = 70.90$, $c_{\text{S1},\text{C3}} = 8.44$
  - $c_{\text{S2},\text{C1}} = 1732.65$, $c_{\text{S2},\text{C2}} = 1780.72$, $c_{\text{S2},\text{C3}} = 567.44$
  - $c_{\text{S3},\text{C1}} = 115.66$, $c_{\text{S3},\text{C2}} = 100.76$, $c_{\text{S3},\text{C3}} = 64.68$

Objective:
\[
\min \left(
    102.33\, y_{\text{S1}} + 94.92\, y_{\text{S2}} + 91.83\, y_{\text{S3}}
    + 1506.22\, x_{\text{S1},\text{C1}} + 70.90\, x_{\text{S1},\text{C2}} + 8.44\, x_{\text{S1},\text{C3}}
    + 1732.65\, x_{\text{S2},\text{C1}} + 1780.72\, x_{\text{S2},\text{C2}} + 567.44\, x_{\text{S2},\text{C3}}
    + 115.66\, x_{\text{S3},\text{C1}} + 100.76\, x_{\text{S3},\text{C2}} + 64.68\, x_{\text{S3},\text{C3}}
\right)
\]

Subject to:

1. Demand satisfaction for each customer:
\[
x_{\text{S1},j} + x_{\text{S2},j} + x_{\text{S3},j} = d_j, \quad \forall j \in \{\text{C1}, \text{C2}, \text{C3}\}
\]
That is,
\[
x_{\text{S1},\text{C1}} + x_{\text{S2},\text{C1}} + x_{\text{S3},\text{C1}} = 1083
\]
\[
x_{\text{S1},\text{C2}} + x_{\text{S2},\text{C2}} + x_{\text{S3},\text{C2}} = 776
\]
\[
x_{\text{S1},\text{C3}} + x_{\text{S2},\text{C3}} + x_{\text{S3},\text{C3}} = 16214
\]

2. Linking constraints (no supply from unopened warehouses):
\[
x_{ij} \leq M_{ij} y_i, \quad \forall i \in \{\text{S1}, \text{S2}, \text{S3}\},\; \forall j \in \{\text{C1}, \text{C2}, \text{C3}\}
\]
where $M_{ij}$ is a sufficiently large constant (e.g., $M_{ij} = d_j$).

3. Variable domains:
\[
y_i \in \{0,1\}, \quad \forall i \in \{\text{S1}, \text{S2}, \text{S3}\}
\]
\[
x_{ij} \geq 0, \quad \forall i \in \{\text{S1}, \text{S2}, \text{S3}\},\; \forall j \in \{\text{C1}, \text{C2}, \text{C3}\}
\]

All parameters and identifiers are as retrieved and preserved in source order.