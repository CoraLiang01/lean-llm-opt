## Mathematical Model

**Indices**  
\(I=\{\,\text{CabinetID}\,\}\) (from *capacity.csv*)  
\(J=\{\,\text{ProductName}\,\}\) (from *products.csv*)

**Parameters**  
\(\text{Capacity}_i\) = capacity of cabinet \(i\) (from *capacity.csv*, column *Capacity*)  
\(\text{Value}_j\) = value per unit of product \(j\) (from *products.csv*, column *Value*)  
\(\text{Weight}_j\) = weight per unit of product \(j\) (from *products.csv*, column *Weight*)

**Decision variables**  
\(x_{ij}\in \mathbb{Z}_{+}\) = number of units of product \(j\) placed in cabinet \(i\)

**Objective**  
\[
\max \;\sum_{i\in I}\sum_{j\in J}\text{Value}_j\,x_{ij}
\]

**Constraints**  

Capacity limits for each cabinet  
\[
\sum_{j\in J}\text{Weight}_j\,x_{ij}\;\le\;\text{Capacity}_i
\qquad \forall\,i\in I
\]

Non‑negativity and integrality  
\[
x_{ij}\;\ge\;0,\quad x_{ij}\in\mathbb{Z}\qquad \forall\,i\in I,\;j\in J
\]

---

## Data Mapping Rules

| Parameter / Variable | Source Table | Source Column | Mapping |
|----------------------|--------------|---------------|---------|
| \(I\) | *capacity.csv* | *CabinetID* | Index set of cabinets |
| \(J\) | *products.csv* | *ProductName* | Index set of products |
| \(\text{Capacity}_i\) | *capacity.csv* | *Capacity* | \(\text{Capacity}_i =\) value in row where *CabinetID* = \(i\) |
| \(\text{Value}_j\) | *products.csv* | *Value* | \(\text{Value}_j =\) value in row where *ProductName* = \(j\) |
| \(\text{Weight}_j\) | *products.csv* | *Weight* | \(\text{Weight}_j =\) value in row where *ProductName* = \(j\) |
| \(x_{ij}\) | Decision variable | – | Integer, non‑negative, defined for all \(i\in I,\;j\in J\) |

All constraints and the objective use only the parameters and variables defined above; no additional data or assumptions are introduced.