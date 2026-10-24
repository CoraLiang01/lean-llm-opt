**Mathematical Model**

Let  

* \(P=\{1,\dots ,80\}\) – set of products.  
* \(D=22\) – number of production days in the month.

| Symbol | Meaning | Domain |
|--------|---------|--------|
| \(x_p\) | Units of product \(p\) produced (100 kg units) | \(\mathbb{Z}_{\ge 0}\) |
| \(y_p\) | 1 if product \(p\) is activated, 0 otherwise | \(\{0,1\}\) |
| \(D_p\) | Maximum demand for product \(p\) (100 kg units) | \(\mathbb{Z}_{>0}\) |
| \(P_p\) | Selling price per 100 kg of product \(p\) | \(\mathbb{R}_{>0}\) |
| \(C_p\) | Production cost per 100 kg of product \(p\) | \(\mathbb{R}_{>0}\) |
| \(Q_p\) | Daily production quota for product \(p\) (100 kg units/day) | \(\mathbb{Z}_{>0}\) |
| \(A_p\) | Fixed activation cost for product \(p\) | \(\mathbb{R}_{>0}\) |
| \(B_p\) | Minimum batch size for product \(p\) (100 kg units) | \(\mathbb{Z}_{>0}\) |

---

### Objective

\[
\max \;\; \sum_{p\in P}\Bigl[(P_p-C_p)\,x_p \;-\; A_p\,y_p\Bigr]
\]

---

### Constraints

1. **Demand limits**  
   \[
   x_p \;\le\; D_p \quad\forall p\in P
   \]

2. **Daily quota limits** (over the whole month)  
   \[
   x_p \;\le\; Q_p\,D \quad\forall p\in P
   \]

3. **Activation linkage** – production only if the line is activated  
   \[
   x_p \;\le\; D_p\,y_p \quad\forall p\in P
   \]

4. **Minimum batch size** – if activated, at least one batch must be produced  
   \[
   x_p \;\ge\; B_p\,y_p \quad\forall p\in P
   \]

5. **Variable domains**  
   \[
   x_p \in \mathbb{Z}_{\ge 0}, \qquad y_p \in \{0,1\}\quad\forall p\in P
   \]

---

**Data Mapping**

| Parameter | Source Table | Source Row | Column |
|-----------|--------------|------------|--------|
| \(D_p\) | `file_0_view_0` | row where `Product = "Maximum Demand (100 kg units)"` | `A1`–`A80` |
| \(P_p\) | `file_0_view_0` | row where `Product = "Selling Price ($/100 kg)"` | `A1`–`A80` |
| \(C_p\) | `file_0_view_0` | row where `Product = "Production Cost ($/100 kg)"` | `A1`–`A80` |
| \(Q_p\) | `file_0_view_0` | row where `Product = "Production Quota (max per day)"` | `A1`–`A80` |
| \(A_p\) | `file_1_view_0` | row where `Product = "Activation Cost ($)"` | `A1`–`A80` |
| \(B_p\) | `file_2_view_0` | row where `Product = "Minimum Batch Size (100 kg units)"` | `A1`–`A80` |

All parameters are read directly from the corresponding columns for each product index \(p\). No additional data or assumptions are introduced beyond what is specified in the query and the source tables.