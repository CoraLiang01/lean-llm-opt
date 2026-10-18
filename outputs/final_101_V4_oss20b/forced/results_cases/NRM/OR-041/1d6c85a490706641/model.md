**Abstract Model**

**Index set**  
\(I = \{\, i \mid i \text{ is an area in New York}\,\}\)

**Parameters**  
\[
\begin{aligned}
&\text{Value}_i &&\text{(benefit per unit of development in area } i\text{)}\\
&\text{Weight}_i &&\text{(resource consumption per unit of development in area } i\text{)}\\
&\text{Capacity} &&\text{(total available development capacity)}
\end{aligned}
\]

**Decision variables**  
\[
x_i \;\ge\; 0 \quad \text{for all } i\in I
\]

**Objective**  
\[
\max \;\sum_{i\in I} \text{Value}_i \cdot x_i
\]

**Constraints**  

1. **Capacity constraint**  
\[
\sum_{i\in I} \text{Weight}_i \cdot x_i \;\le\; \text{Capacity}
\]

---

**Data Mapping**

| Parameter | Source table_id | Source column |
|-----------|-----------------|---------------|
| Capacity  | `file_0_view_0` | `Capacity`    |
| Value_i   | `file_1_view_0` | `Value`       |
| Weight_i  | `file_1_view_0` | `Weight`      |
| I         | `file_1_view_0` | `ProductName` |

*(No record values or counts are reproduced.)*