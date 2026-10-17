**Mathematical Model**

Let  

* \(I=\{S1,S2\}\) – set of suppliers  
* \(J=\{C1,C2\}\) – set of supermarkets  

Parameters  

\[
\begin{aligned}
f_i &\;=\;\text{fixed cost of supplier }i &&\forall i\in I\\
d_j &\;=\;\text{demand of supermarket }j &&\forall j\in J\\
c_{ij}&=\;\text{unit transportation cost from supplier }i\text{ to supermarket }j &&\forall i\in I,\;j\in J
\end{aligned}
\]

Decision variables  

\[
\begin{aligned}
y_i &\in\{0,1\} &&\text{(supplier }i\text{ is activated)}\\
x_{ij}&\ge 0 &&\text{(units shipped from }i\text{ to }j)
\end{aligned}
\]

Let \(M=\sum_{j\in J} d_j\) (a sufficiently large constant).

**Objective**

\[
\min \;\sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

**Constraints**

1. **Demand satisfaction**  
   \[
   \sum_{i\in I} x_{ij}\;\ge\; d_j \qquad \forall j\in J
   \]

2. **Linking activation and shipment**  
   \[
   x_{ij}\;\le\; M\,y_i \qquad \forall i\in I,\;j\in J
   \]

3. **Non‑negativity**  
   \[
   x_{ij}\;\ge\;0 \qquad \forall i\in I,\;j\in J
   \]

---

### Data Mapping Rules

| Parameter | Source Table | Key Column | Value Column |
|-----------|--------------|------------|--------------|
| \(f_i\)   | `file_1_view_0` | `Unnamed: 0` (supplier id) | `fixed_costs` |
| \(d_j\)   | `file_0_view_0` | `customer` (supermarket id) | `demand` |
| \(c_{ij}\)| `file_2_view_0` | `Unnamed: 0` (supplier id) | `C1`, `C2` (supermarket columns) |

*All indices \(i\) and \(j\) are taken from the distinct values of the key columns in the respective tables.*