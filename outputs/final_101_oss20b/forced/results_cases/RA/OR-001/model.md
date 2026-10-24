**Mathematical Model**

Let  

- \(S\) be the set of suppliers (distribution centers).  
- \(C\) be the set of customers (customer groups).  

Decision variable  
\[
x_{s c}\;\;(\text{units shipped from supplier }s\text{ to customer }c),\qquad x_{s c}\ge 0.
\]

Parameters  

\[
\begin{aligned}
\text{demand}_c &\ge 0 &&\text{(daily demand of customer }c),\\
\text{supply}_s &\ge 0 &&\text{(daily supply capacity of supplier }s),\\
\text{cost}_{s c} &\ge 0 &&\text{(transportation cost per unit from }s\text{ to }c).
\end{aligned}
\]

**Objective**

\[
\min \;\sum_{s\in S}\sum_{c\in C}\text{cost}_{s c}\;x_{s c}
\]

**Constraints**

1. **Demand satisfaction**  
   \[
   \sum_{s\in S}x_{s c}\;\ge\;\text{demand}_c,\qquad \forall c\in C
   \]

2. **Supply capacity**  
   \[
   \sum_{c\in C}x_{s c}\;\le\;\text{supply}_s,\qquad \forall s\in S
   \]

3. **Non‑negativity**  
   \[
   x_{s c}\;\ge\;0,\qquad \forall s\in S,\;c\in C
   \]

---

### Data Mapping

| Parameter | Source Table | Key Column | Value Column | Mapping Expression |
|-----------|--------------|------------|--------------|--------------------|
| \(\text{demand}_c\) | `file_0_view_0` | `customer` | `demand` | `demand[c] = file_0_view_0[customer = c][demand]` |
| \(\text{supply}_s\) | `file_1_view_0` | `Unnamed: 0` | `supply_capacity` | `supply[s] = file_1_view_0[Unnamed: 0 = s][supply_capacity]` |
| \(\text{cost}_{s c}\) | `file_2_view_0` | `Unnamed: 0` | `C1`–`C18` | `cost[s,c] = file_2_view_0[Unnamed: 0 = s][C{c}]` |

*Note:* `C{c}` denotes the column whose name matches the customer identifier (e.g., `C1` for customer `C1`). All indices `s` and `c` are taken from the corresponding key columns in the source tables.