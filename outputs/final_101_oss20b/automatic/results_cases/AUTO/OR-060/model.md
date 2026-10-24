**Mathematical Model**

Let  

* \(I=\{S_1,\dots ,S_{12}\}\) – set of suppliers  
* \(J=\{C_1,\dots ,C_{12}\}\) – set of supermarkets  

Parameters (all obtained from the CSV files)

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(d_j\) | Demand of supermarket \(j\) | `file_0_view_0` column **demand** |
| \(f_i\) | Fixed opening cost of supplier \(i\) | `file_1_view_0` column **fixed_costs** |
| \(c_{ij}\) | Transportation cost per unit from supplier \(i\) to supermarket \(j\) | `file_2_view_0` columns **C1**–**C12** |
| \(M\) | Big‑\(M\) upper bound on total shipment from a supplier | \(M=\sum_{j\in J} d_j\) (computed from the demand data) |

Decision variables  

\[
\begin{aligned}
x_{ij} &\ge 0 &&\text{quantity shipped from supplier }i\text{ to supermarket }j,\\
y_i &\in\{0,1\} &&\text{1 if supplier }i\text{ is opened, 0 otherwise.}
\end{aligned}
\]

**Objective**

\[
\min \;\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}\;+\;\sum_{i\in I} f_i\,y_i
\]

**Constraints**

1. **Demand satisfaction** – each supermarket receives exactly its demand  

\[
\sum_{i\in I} x_{ij}=d_j ,\qquad \forall j\in J
\]

2. **Supplier activation** – a supplier can ship only if it is opened  

\[
\sum_{j\in J} x_{ij}\;\le\; M\,y_i ,\qquad \forall i\in I
\]

3. **Variable domains**  

\[
x_{ij}\ge 0,\; y_i\in\{0,1\}\quad \forall i\in I,\; j\in J
\]

---

### Data Mapping

| Parameter | Table ID | Column(s) |
|-----------|----------|-----------|
| \(d_j\) | `file_0_view_0` | `demand` |
| \(f_i\) | `file_1_view_0` | `fixed_costs` |
| \(c_{ij}\) | `file_2_view_0` | `C1`–`C12` (each column corresponds to a supermarket) |
| \(M\) | – | computed as \(\sum_{j} d_j\) from `file_0_view_0` |

All indices \(i\) and \(j\) are taken from the row and column identifiers in the respective tables (suppliers `S1`–`S12`, supermarkets `C1`–`C12`). No additional capacity limits are imposed beyond the big‑\(M\) constraint.