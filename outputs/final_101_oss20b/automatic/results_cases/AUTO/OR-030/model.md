**Mathematical Model**

```markdown
**Sets**  
\(I\) – set of car models (all rows with product name starting with “FDK57”).

**Parameters**  
\(R_i\) – revenue per unit of model \(i\).  
\(D_i\) – deterministic demand for model \(i\).  
\(I_i\) – initial inventory of model \(i\).

**Decision Variables**  
\(x_i \in \mathbb{Z}_+\) – quantity of model \(i\) the dealership plans to sell.

**Objective**  
\[
\max \;\sum_{i\in I} R_i\,x_i
\]

**Constraints**  

1. **Demand limit**  
\[
x_i \;\le\; D_i , \qquad \forall i\in I
\]

2. **Inventory limit**  
\[
x_i \;\le\; I_i , \qquad \forall i\in I
\]

3. **Non‑negativity / integrality**  
\[
x_i \in \mathbb{Z}_+ , \qquad \forall i\in I
\]
```

---

**Data Mapping**

| Source | Table ID | Column | Parameter / Index |
|--------|----------|--------|-------------------|
| BigMartSales.csv (filtered view) | `file_0_view_0` | `Product Name` | index \(i\) |
| | | `Revenue` | \(R_i\) |
| | | `Demand` | \(D_i\) |
| | | `Initial Inventory` | \(I_i\) |

*Each row in `file_0_view_0` defines a distinct element of \(I\) and supplies the corresponding parameter values.*