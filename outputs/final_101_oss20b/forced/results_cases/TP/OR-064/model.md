**Mathematical model**

\[
\begin{aligned}
\min\;&\sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{s.t.}\;&\sum_{i\in I} x_{ij}=d_j \qquad &&\forall j\in J \quad\text{(demand)}\\[2pt]
&x_{ij}\;\le\;d_j\,y_i \qquad &&\forall i\in I,\;j\in J \quad\text{(link)}\\[2pt]
&x_{ij}\;\ge\;0 \qquad &&\forall i\in I,\;j\in J\\
&y_i\in\{0,1\} \qquad &&\forall i\in I
\end{aligned}
\]

**Index sets**

* \(I=\{S1,S2,\dots ,S24\}\) (suppliers, taken from the identifier column **Unnamed: 0** of *fixed_cost.csv* and *transportation_costs.csv*)
* \(J=\{C1,C2,\dots ,C25\}\) (customers, taken from the identifier column **customer** of *demand.csv* and the column headers of *transportation_costs.csv*)

**Parameters**

| Symbol | Meaning | Source table & column |
|--------|---------|-----------------------|
| \(d_j\) | Demand of customer \(j\) | *file_0_view_0*, column **demand** |
| \(f_i\) | Fixed opening cost of supplier \(i\) | *file_1_view_0*, column **fixed_costs** |
| \(c_{ij}\) | Transportation cost per unit from supplier \(i\) to customer \(j\) | *file_2_view_0*, columns **C1**–**C25** |

**Decision variables**

* \(y_i\in\{0,1\}\) – 1 if supplier \(i\) is opened, 0 otherwise.
* \(x_{ij}\ge 0\) – quantity shipped from supplier \(i\) to customer \(j\).

---

### Data‑mapping rules

| Parameter | Table ID | Column name | Index column |
|-----------|----------|-------------|--------------|
| \(d_j\) | `file_0_view_0` | `demand` | `customer` |
| \(f_i\) | `file_1_view_0` | `fixed_costs` | `Unnamed: 0` |
| \(c_{ij}\) | `file_2_view_0` | `C1`–`C25` | `Unnamed: 0` (rows) and column headers (customers) |

These mappings ensure that every coefficient used in the model is directly traceable to the corresponding source file and column, preserving the exact identifiers and values as supplied.