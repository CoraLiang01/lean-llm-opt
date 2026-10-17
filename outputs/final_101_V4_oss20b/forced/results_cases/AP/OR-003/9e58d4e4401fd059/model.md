**Mathematical Model**

Let  

- \(S\) be the set of suppliers (indices \(s\)).  
- \(C\) be the set of customer groups (indices \(c\)).  

Parameters  

\[
\begin{aligned}
\text{SupplyCap}_{s} &\quad\text{: daily supply capacity of supplier }s,\\
\text{Demand}_{c} &\quad\text{: daily demand of customer group }c,\\
\text{Cost}_{s,c} &\quad\text{: transportation cost per unit from supplier }s\text{ to customer }c.
\end{aligned}
\]

Decision variables  

\[
x_{s,c}\;\;(\text{units shipped from }s\text{ to }c),\qquad x_{s,c}\ge 0.
\]

---

**Objective**

\[
\min \;\sum_{s\in S}\sum_{c\in C}\text{Cost}_{s,c}\;x_{s,c}
\]

---

**Constraints**

1. **Supply capacity**

\[
\sum_{c\in C} x_{s,c}\;\le\;\text{SupplyCap}_{s}\qquad \forall s\in S
\]

2. **Demand satisfaction**

\[
\sum_{s\in S} x_{s,c}\;=\;\text{Demand}_{c}\qquad \forall c\in C
\]

3. **Non‑negativity**

\[
x_{s,c}\;\ge\;0\qquad \forall s\in S,\;c\in C
\]

---

**Data Mapping**

| Parameter | Source Table | Key Column | Value Column |
|-----------|--------------|------------|--------------|
| \(\text{SupplyCap}_{s}\) | `file_1_view_0` | `Unnamed: 0` | `supply_capacity` |
| \(\text{Demand}_{c}\) | `file_0_view_0` | `customer` | `demand` |
| \(\text{Cost}_{s,c}\) | `file_2_view_0` | `Unnamed: 0` | `C1`–`C10` (one column per customer) |

*All indices \(s\) and \(c\) are taken from the distinct values of the key columns in the respective tables.*