**Mathematical Model**

Let  

* \(P=\{1,\dots ,80\}\) – product indices (each index corresponds to the column \(A_p\) in the CSV files).  
* \(D=\{1,\dots ,22\}\) – production days.

**Decision variables**

\[
\begin{aligned}
q_{p,d} &\in \mathbb{Z}_{\ge 0} &&\text{units of product }p\text{ produced on day }d,\\
z_{p,d} &\in \{0,1\} &&\text{production indicator on day }d,\\
y_{p}   &\in \{0,1\} &&\text{activation indicator for product line }p.
\end{aligned}
\]

**Parameters** (all taken from the CSV files – see Data Mapping below)

\[
\begin{aligned}
\text{Demand}_p      &\ge 0,\\
\text{Price}_p       &\ge 0,\\
\text{Cost}_p        &\ge 0,\\
\text{Quota}_p       &\ge 0,\\
\text{ActCost}_p     &\ge 0,\\
\text{Batch}_p       &\ge 0.
\end{aligned}
\]

**Objective**

\[
\max \;\sum_{p\in P}\sum_{d\in D}\bigl(\text{Price}_p-\text{Cost}_p\bigr)\,q_{p,d}
      \;-\;\sum_{p\in P}\text{ActCost}_p\,y_{p}
\]

**Constraints**

1. **Demand satisfaction**  
   \[
   \sum_{d\in D} q_{p,d}\;\le\;\text{Demand}_p \qquad \forall p\in P
   \]

2. **Daily quota**  
   \[
   q_{p,d}\;\le\;\text{Quota}_p\,z_{p,d} \qquad \forall p\in P,\;d\in D
   \]

3. **Minimum batch size**  
   \[
   q_{p,d}\;\ge\;\text{Batch}_p\,z_{p,d} \qquad \forall p\in P,\;d\in D
   \]

4. **Activation linkage**  
   \[
   y_{p}\;\ge\;z_{p,d} \qquad \forall p\in P,\;d\in D
   \]

5. **Activation consistency**  
   \[
   y_{p}\;\le\;\sum_{d\in D} z_{p,d} \qquad \forall p\in P
   \]

6. **Variable domains** (already stated in the variable definitions).

---

## Data Mapping

| Parameter | Table ID | Row | Column | Description |
|-----------|----------|-----|--------|-------------|
| `Demand_p` | `file_0_view_0` | 0 | `A{p}` | Maximum demand (100 kg units) |
| `Price_p`  | `file_0_view_0` | 1 | `A{p}` | Selling price ($/100 kg) |
| `Cost_p`   | `file_0_view_0` | 2 | `A{p}` | Production cost ($/100 kg) |
| `Quota_p`  | `file_0_view_0` | 3 | `A{p}` | Production quota (max per day) |
| `ActCost_p`| `file_1_view_0` | 0 | `A{p}` | Activation cost ($) |
| `Batch_p`  | `file_2_view_0` | 0 | `A{p}` | Minimum batch size (100 kg units) |

*All indices \(p\) refer to the same column \(A_p\) across the tables.*