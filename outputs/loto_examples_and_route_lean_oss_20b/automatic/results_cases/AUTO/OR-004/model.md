**Mathematical Model**

Let  

\[
S=\{s\mid s\text{ is a supply center}\},\qquad  
C=\{c\mid c\text{ is a customer group}\}
\]

Decision variables  

\[
x_{s,c}\in \mathbb{Z}_{\ge 0}\qquad \forall s\in S,\;c\in C
\]

Parameters  

\[
\begin{aligned}
\text{demand}_c &\ge 0 &&\forall c\in C\\
\text{supply}_s &\ge 0 &&\forall s\in S\\
\text{cost}_{s,c} &\ge 0 &&\forall s\in S,\;c\in C
\end{aligned}
\]

Objective  

\[
\min \;\sum_{s\in S}\sum_{c\in C}\text{cost}_{s,c}\,x_{s,c}
\]

Constraints  

\[
\begin{aligned}
\text{Demand satisfaction:}\quad &\sum_{s\in S}x_{s,c}\;\ge\;\text{demand}_c &&\forall c\in C\\[4pt]
\text{Supply capacity:}\quad &\sum_{c\in C}x_{s,c}\;\le\;\text{supply}_s &&\forall s\in S
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Table ID | Column(s) | Mapping |
|-----------|----------|-----------|---------|
| `demand_c` | `file_0_view_0` | `customer`, `demand` | `demand_c = demand` where row id = `customer` |
| `supply_s` | `file_1_view_0` | `Unnamed: 0`, `supply_capacity` | `supply_s = supply_capacity` where row id = `Unnamed: 0` |
| `cost_{s,c}` | `file_2_view_0` | matrix `Unnamed: 0` × `customer` columns | `cost_{s,c} = value` where row id = `Unnamed: 0`, column id = `customer` |

| Variable | Domain | Notes |
|----------|--------|-------|
| `x_{s,c}` | Integer, \(x_{s,c}\ge 0\) | Whole units of product shipped |

All index sets `S` and `C` are derived from the row identifiers of `file_1_view_0` and `file_0_view_0`, respectively. The cost matrix is taken directly from `file_2_view_0` with row and column identifiers matching the supply and customer identifiers.