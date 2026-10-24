**Mathematical Model**

Let  

* \(S=\{s\mid s\text{ is a distribution center}\}\)  
* \(C=\{c\mid c\text{ is a customer group}\}\)

**Parameters**

\[
\begin{aligned}
\text{demand}[c] &\quad\text{(units required by customer }c)\\
\text{supply}[s] &\quad\text{(maximum units that center }s\text{ can ship)}\\
\text{cost}[s,c] &\quad\text{(transportation cost per unit from }s\text{ to }c)
\end{aligned}
\]

**Decision Variables**

\[
x_{s,c}\in\mathbb{Z}_{\ge 0}\qquad\forall s\in S,\;c\in C
\]

**Objective**

\[
\min \;\sum_{s\in S}\sum_{c\in C}\text{cost}[s,c]\;x_{s,c}
\]

**Constraints**

1. **Supply capacity**

\[
\sum_{c\in C}x_{s,c}\;\le\;\text{supply}[s]\qquad\forall s\in S
\]

2. **Demand satisfaction**

\[
\sum_{s\in S}x_{s,c}\;\ge\;\text{demand}[c]\qquad\forall c\in C
\]

---

### Data Mapping

| Parameter | Table ID | Key Column | Value Column(s) | Notes |
|-----------|----------|------------|-----------------|-------|
| `demand[c]` | `file_0_view_0` | `customer` | `demand` | One row per customer |
| `supply[s]` | `file_1_view_0` | `Unnamed: 0` | `supply_capacity` | One row per distribution center |
| `cost[s,c]` | `file_2_view_0` | `Unnamed: 0` | `C1`, `C2`, `C3`, `C4` | For each center `s`, the cost to customer `c` is taken from the column whose name equals `c`.  Costs are defined only for customers `C1`–`C4`; for other customers the parameter is undefined. |

*All indices (`S` and `C`) are taken from the full set of rows returned in the observation.*