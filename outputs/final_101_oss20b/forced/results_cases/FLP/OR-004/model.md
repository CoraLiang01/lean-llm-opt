**Mathematical Model**

Let  

* \(I=\{S1,\dots ,S12\}\) – set of distribution centers (suppliers)  
* \(J=\{C1,\dots ,C12\}\) – set of customer groups  

Parameters  

\[
\begin{aligned}
d_j &\ge 0 &&\text{demand of customer }j \\
s_i &\ge 0 &&\text{supply capacity of center }i \\
c_{ij} &\ge 0 &&\text{transportation cost per unit from }i\text{ to }j
\end{aligned}
\]

Decision variables  

\[
x_{ij}\;\ge 0 \qquad \text{units shipped from }i\text{ to }j
\]

Objective  

\[
\min \;\sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

Constraints  

1. **Demand satisfaction**  
   \[
   \sum_{i\in I} x_{ij}=d_j \qquad \forall j\in J
   \]

2. **Supply capacity**  
   \[
   \sum_{j\in J} x_{ij}\le s_i \qquad \forall i\in I
   \]

All variables are continuous and non‑negative.

---

**Data Mapping**

| Parameter | Table ID | Row ID column | Column ID column | Value column |
|-----------|----------|---------------|------------------|--------------|
| \(d_j\)   | `file_0_view_0` | `customer` | – | `demand` |
| \(s_i\)   | `file_1_view_0` | `Unnamed: 0` | – | `supply_capacity` |
| \(c_{ij}\)| `file_2_view_0` | `Unnamed: 0` | `customer` | cost value (matrix entry) |

The index sets \(I\) and \(J\) are the distinct values appearing in the row‑ID and column‑ID columns of the respective tables. No other parameters or constraints are required.