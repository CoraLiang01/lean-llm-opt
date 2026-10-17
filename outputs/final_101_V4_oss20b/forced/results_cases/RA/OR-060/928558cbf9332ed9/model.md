**Mathematical Model**

Let  

* \(S\) – set of suppliers (indices from *fixed_cost.csv*).  
* \(C\) – set of customers (indices from *demand.csv*).  

Parameters  

\[
\begin{aligned}
&\text{fixed\_cost}[s] &&\text{(fixed cost of opening supplier }s)\\
&\text{demand}[c] &&\text{(demand of customer }c)\\
&\text{transport}[s,c] &&\text{(transport cost per unit from supplier }s\text{ to customer }c)
\end{aligned}
\]

Decision variables  

\[
\begin{aligned}
&y_s &\in\{0,1\} &&\text{(supplier }s\text{ is opened)}\\
&x_{s,c} &\ge 0 &&\text{(units shipped from }s\text{ to }c)
\end{aligned}
\]

---

\[
\begin{aligned}
\min\;&\sum_{s\in S}\text{fixed\_cost}[s]\;y_s
      \;+\;\sum_{s\in S}\sum_{c\in C}\text{transport}[s,c]\;x_{s,c} \\[4pt]
\text{s.t.}\;&\sum_{s\in S}x_{s,c}= \text{demand}[c] &&\forall\,c\in C \quad\text{(meet demand)}\\[4pt]
&x_{s,c}\;\le\;\text{demand}[c]\;y_s &&\forall\,s\in S,\;c\in C \quad\text{(linking)}\\[4pt]
&y_s\in\{0,1\} &&\forall\,s\in S\\
&x_{s,c}\ge 0 &&\forall\,s\in S,\;c\in C
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Table ID | Key Column | Value Column |
|-----------|----------|------------|--------------|
| `fixed_cost[s]` | `file_1_view_0` | `Unnamed: 0` | `fixed_costs` |
| `demand[c]` | `file_0_view_0` | `customer` | `demand` |
| `transport[s,c]` | `file_2_view_0` | `Unnamed: 0` (row) | column named `C1`–`C12` (customer) |

*Index sets*  
- \(S\) is the set of distinct values in `Unnamed: 0` of `file_1_view_0`.  
- \(C\) is the set of distinct values in `customer` of `file_0_view_0`.  

All parameters are read directly from the corresponding CSV files; no additional data are invented.