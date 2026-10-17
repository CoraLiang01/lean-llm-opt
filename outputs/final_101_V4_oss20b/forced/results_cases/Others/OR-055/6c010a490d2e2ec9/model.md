## Mathematical Model

**Sets**  
\(I = \{\,\text{DisplayID}\,\}\) (from *capacity.csv*)  
\(J = \{\,\text{ProductName}\,\}\) (from *products.csv*)

**Parameters**  
\(C_i =\) Capacity of display \(i\) (from *capacity.csv*, column *Capacity*)  
\(V_j =\) Value of product \(j\) (from *products.csv*, column *Value*)  
\(W_j =\) Weight of product \(j\) (from *products.csv*, column *Weight*)

**Decision Variables**  
\(x_{ij}\in \mathbb{Z}_{+}\) number of units of product \(j\) placed in display \(i\)

**Objective**  
\[
\max \;\sum_{i\in I}\sum_{j\in J} V_j\,x_{ij}
\]

**Constraints**  
Capacity of each display  
\[
\sum_{j\in J} W_j\,x_{ij}\;\le\; C_i \qquad \forall\, i\in I
\]

All variables are non‑negative integers:
\[
x_{ij}\;\ge\;0,\; x_{ij}\in\mathbb{Z}\qquad \forall\, i\in I,\; j\in J
\]

---

## Data Mapping Rules

| Parameter / Variable | Source Table | Source Column |
|----------------------|--------------|---------------|
| \(I\) | *capacity.csv* | *DisplayID* |
| \(J\) | *products.csv* | *ProductName* |
| \(C_i\) | *capacity.csv* | *Capacity* |
| \(V_j\) | *products.csv* | *Value* |
| \(W_j\) | *products.csv* | *Weight* |
| \(x_{ij}\) | Decision variable | – |

All indices \(i\) and \(j\) are taken from the full set of rows returned by the respective source files. No additional filtering or enumeration is applied.