**Abstract Model**

**Index Sets**  
\(I\) – set of platforms (identified by PlatformID in *capacity.csv*)  
\(J\) – set of products (identified by ProductName in *products.csv*)

**Parameters**  
\(C_i\) – memory capacity of platform \(i\) (value of column *Capacity* in *capacity.csv*)  
\(V_j\) – value of product \(j\) (value of column *Value* in *products.csv*)  
\(W_j\) – memory requirement of product \(j\) (value of column *Weight* in *products.csv*)

**Decision Variables**  
\(x_{ij}\) – number of units of product \(j\) to list on platform \(i\)  
\(x_{ij}\in \mathbb{Z}_{+}\) for all \(i\in I,\;j\in J\)

**Objective Function**  
\[
\max \sum_{i\in I}\sum_{j\in J} V_j \, x_{ij}
\]

**Constraints**  

1. **Platform capacity constraints**  
\[
\sum_{j\in J} W_j \, x_{ij} \;\le\; C_i , \qquad \forall i\in I
\]

2. **Non‑negativity and integrality**  
\[
x_{ij}\in \mathbb{Z}_{+}, \qquad \forall i\in I,\;j\in J
\]

---

**Data Mapping**

| Parameter | Source Table | Exact Column Name |
|-----------|--------------|-------------------|
| \(C_i\)   | *capacity.csv* | *Capacity* |
| \(V_j\)   | *products.csv* | *Value* |
| \(W_j\)   | *products.csv* | *Weight* |

The index sets \(I\) and \(J\) are derived from the unique values of *PlatformID* and *ProductName* in the respective tables. No record values or counts are reproduced.