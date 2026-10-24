**Mathematical Model**

Let  

- \(S\) be the set of suppliers (identified by the values in column **`Unnamed: 0`** of *file_1_view_0* and *file_2_view_0*).  
- \(C\) be the set of customers (identified by the values in column **`customer`** of *file_0_view_0*).  

Decision variables  

\[
x_{s c}\in \mathbb{Z}_{\ge 0}\qquad \forall s\in S,\;c\in C
\]

(Units of product shipped from supplier \(s\) to customer \(c\).)

Parameters  

\[
\begin{aligned}
\text{supply}_s &\;=\; \text{value of column `supply_capacity` in row `Unnamed: 0` = }s \text{ of *file_1_view_0*} \\
\text{demand}_c &\;=\; \text{value of column `demand` in row `customer` = }c \text{ of *file_0_view_0*} \\
\text{cost}_{s c} &\;=\; \text{value in row `Unnamed: 0` = }s \text{ and column `C?` = }c \text{ of *file_2_view_0*}
\end{aligned}
\]

Objective  

\[
\min \;\sum_{s\in S}\sum_{c\in C}\text{cost}_{s c}\;x_{s c}
\]

Supply‑capacity constraints  

\[
\sum_{c\in C}x_{s c}\;\le\;\text{supply}_s \qquad \forall s\in S
\]

Demand‑satisfaction constraints  

\[
\sum_{s\in S}x_{s c}\;=\;\text{demand}_c \qquad \forall c\in C
\]

Non‑negativity and integrality  

\[
x_{s c}\;\in\;\mathbb{Z}_{\ge 0}\qquad \forall s\in S,\;c\in C
\]

---

**Data Mapping**

| Parameter | Table ID | Row key column | Value column |
|-----------|----------|----------------|--------------|
| `supply_s` | `file_1_view_0` | `Unnamed: 0` | `supply_capacity` |
| `demand_c` | `file_0_view_0` | `customer` | `demand` |
| `cost_{s c}` | `file_2_view_0` | `Unnamed: 0` | `C?` (where `?` is the customer identifier, e.g., `C1`, `C2`, …, `C10`) |

*All index sets \(S\) and \(C\) are taken directly from the row identifiers of the corresponding tables.*