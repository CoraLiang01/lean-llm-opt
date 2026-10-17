**Mathematical Model**

Let  

* \(P=\{1,\dots ,80\}\) – set of products  
* \(D=\{1,\dots ,22\}\) – set of days in the month  

**Parameters**

| Symbol | Meaning | Source (table_id, row, column) |
|--------|---------|--------------------------------|
| \(D_{\max }(p)\) | Maximum demand (100 kg units) | `file_0_view_0`, row 0, column `A{p}` |
| \(P_{\text{price}}(p)\) | Selling price ($/100 kg) | `file_0_view_0`, row 1, column `A{p}` |
| \(P_{\text{cost}}(p)\) | Production cost ($/100 kg) | `file_0_view_0`, row 2, column `A{p}` |
| \(Q(p)\) | Production quota (max per day) | `file_0_view_0`, row 3, column `A{p}` |
| \(B(p)\) | Minimum batch size (100 kg units) | `file_2_view_0`, row 0, column `A{p}` |
| \(C_{\text{act}}(p)\) | Activation cost ($) | `file_1_view_0`, row 0, column `A{p}` |

**Decision Variables**

* \(q_{p,d}\in \mathbb{Z}_{+}\) – quantity of product \(p\) produced on day \(d\) (units of 100 kg)  
* \(z_{p}\in\{0,1\}\) – 1 if product \(p\) is produced at least one day, 0 otherwise  

**Objective**

\[
\max \;\sum_{p\in P}\sum_{d\in D}
q_{p,d}\,\bigl(P_{\text{price}}(p)-P_{\text{cost}}(p)\bigr)
-\sum_{p\in P} C_{\text{act}}(p)\,z_{p}
\]

**Constraints**

1. **Daily production quota**  
   \[
   q_{p,d}\;\le\; Q(p)\,z_{p}\qquad \forall p\in P,\; d\in D
   \]

2. **Demand satisfaction**  
   \[
   \sum_{d\in D} q_{p,d}\;\le\; D_{\max }(p)\qquad \forall p\in P
   \]

3. **Minimum batch size**  
   \[
   \sum_{d\in D} q_{p,d}\;\ge\; B(p)\,z_{p}\qquad \forall p\in P
   \]

4. **Variable domains**  
   \[
   q_{p,d}\in\mathbb{Z}_{+},\qquad z_{p}\in\{0,1\}\qquad \forall p\in P,\; d\in D
   \]

---

**Data Mapping**

| Parameter | Table ID | Row | Column |
|-----------|----------|-----|--------|
| \(D_{\max }(p)\) | `file_0_view_0` | 0 | `A{p}` |
| \(P_{\text{price}}(p)\) | `file_0_view_0` | 1 | `A{p}` |
| \(P_{\text{cost}}(p)\) | `file_0_view_0` | 2 | `A{p}` |
| \(Q(p)\) | `file_0_view_0` | 3 | `A{p}` |
| \(B(p)\) | `file_2_view_0` | 0 | `A{p}` |
| \(C_{\text{act}}(p)\) | `file_1_view_0` | 0 | `A{p}` |

*All indices \(p\) run from 1 to 80, and \(d\) from 1 to 22.*