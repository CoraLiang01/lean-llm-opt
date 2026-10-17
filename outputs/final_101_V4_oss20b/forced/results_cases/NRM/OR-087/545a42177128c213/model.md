**Mathematical Model**

Let  

* \(P=\{1,\dots ,80\}\) – set of products (columns \(A1\)–\(A80\)).  
* \(D=\{1,\dots ,22\}\) – set of days in the month.

**Parameters** (all indexed by \(p\in P\))

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(D_p\) | Maximum demand (100 kg units) | `file_0_view_0`, row “Maximum Demand (100 kg units)”, column `A{p}` |
| \(P_p\) | Selling price ($/100 kg) | `file_0_view_0`, row “Selling Price ($/100 kg)”, column `A{p}` |
| \(C_p\) | Production cost ($/100 kg) | `file_0_view_0`, row “Production Cost ($/100 kg)”, column `A{p}` |
| \(Q_p\) | Daily production quota (max units per day if line dedicated) | `file_0_view_0`, row “Production Quota (max per day)”, column `A{p}` |
| \(AC_p\) | Fixed activation cost ($) | `file_1_view_0`, row “Activation Cost ($)”, column `A{p}` |
| \(MB_p\) | Minimum batch size (100 kg units) | `file_2_view_0`, row “Minimum Batch Size (100 kg units)”, column `A{p}` |

**Decision Variables**

* \(q_{p,d}\in \mathbb{Z}_{+}\) – quantity of product \(p\) produced on day \(d\) (units of 100 kg).  
* \(y_p\in\{0,1\}\) – 1 if production line for product \(p\) is activated, 0 otherwise.

**Objective**

\[
\max \;\sum_{p\in P}\sum_{d\in D}\bigl(P_p-C_p\bigr)\,q_{p,d}\;-\;\sum_{p\in P} AC_p\,y_p
\]

**Constraints**

1. **Demand limits**  
   \[
   \sum_{d\in D} q_{p,d}\;\le\; D_p \qquad \forall p\in P
   \]

2. **Daily quota (link to activation)**  
   \[
   q_{p,d}\;\le\; Q_p\,y_p \qquad \forall p\in P,\; d\in D
   \]

3. **Minimum batch size**  
   \[
   \sum_{d\in D} q_{p,d}\;\ge\; MB_p\,y_p \qquad \forall p\in P
   \]

4. **Activation binary**  
   \[
   y_p\in\{0,1\}\qquad \forall p\in P
   \]

5. **Non‑negativity and integrality**  
   \[
   q_{p,d}\in\mathbb{Z}_{+}\qquad \forall p\in P,\; d\in D
   \]

---

### Data Mapping

| Parameter | Table ID | Row (label) | Column |
|-----------|----------|-------------|--------|
| \(D_p\) | `file_0_view_0` | “Maximum Demand (100 kg units)” | `A{p}` |
| \(P_p\) | `file_0_view_0` | “Selling Price ($/100 kg)” | `A{p}` |
| \(C_p\) | `file_0_view_0` | “Production Cost ($/100 kg)” | `A{p}` |
| \(Q_p\) | `file_0_view_0` | “Production Quota (max per day)” | `A{p}` |
| \(AC_p\) | `file_1_view_0` | “Activation Cost ($)” | `A{p}` |
| \(MB_p\) | `file_2_view_0` | “Minimum Batch Size (100 kg units)” | `A{p}` |

All indices \(p\) run over the 80 products, and all days \(d\) run over the 22 days. No additional data or assumptions are introduced beyond what is specified in the query and the provided CSV files.