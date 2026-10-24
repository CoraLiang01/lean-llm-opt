## Mathematical Model

**Index set**  
\(P=\{p:\text{product name in 41.csv}\}\)

**Parameters**  
\[
\begin{aligned}
&\text{Labor}_p &&\leftarrow \text{Labor per unit of product }p \\
&\text{Material}_p &&\leftarrow \text{Material per unit of product }p \\
&\text{Price}_p &&\leftarrow \text{Selling Price of product }p \\
&\text{VarCost}_p &&\leftarrow \text{Variable Cost per unit of product }p
\end{aligned}
\]

**Decision variables**  
\[
x_p \ge 0 \quad \forall p\in P
\]

**Objective**  
\[
\max \;\; \sum_{p\in P}\bigl(\text{Price}_p-\text{VarCost}_p\bigr)\,x_p \;-\; 4500
\]

**Constraints**  

Labor capacity  
\[
\sum_{p\in P}\text{Labor}_p\,x_p \;\le\; 1650
\]

Material capacity  
\[
\sum_{p\in P}\text{Material}_p\,x_p \;\le\; 1850
\]

Non‑negativity  
\[
x_p \ge 0 \quad \forall p\in P
\]

---

## Data Mapping

| Parameter | Source Table | Column Name |
|-----------|--------------|-------------|
| \(\text{Labor}_p\) | `file_0_view_0` | `Labor per unit` |
| \(\text{Material}_p\) | `file_0_view_0` | `Material per unit` |
| \(\text{Price}_p\) | `file_0_view_0` | `Selling Price` |
| \(\text{VarCost}_p\) | `file_0_view_0` | `Variable Cost` |

The index set \(P\) consists of all distinct values in the `Product Name` column of `file_0_view_0`. The fixed operating cost of \$4,500 is a constant in the objective. No other data are required.