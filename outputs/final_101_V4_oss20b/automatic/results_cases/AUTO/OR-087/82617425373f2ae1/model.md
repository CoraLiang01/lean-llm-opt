## Mathematical Model

Let  

* \(I=\{1,\dots ,80\}\) – set of products.  
* \(x_i\in \mathbb Z_{\ge 0}\) – quantity of product \(i\) produced (units of 100 kg).  
* \(y_i\in\{0,1\}\) – 1 if a production line is activated for product \(i\), 0 otherwise.  

Parameters (all taken directly from the CSV files):

| Parameter | Symbol | Source | Column |
|-----------|--------|--------|--------|
| Maximum demand | \(d_i\) | 36‑1.csv | “Maximum Demand (100 kg units)” |
| Selling price | \(p_i\) | 36‑1.csv | “Selling Price ($/100 kg)” |
| Production cost | \(c_i\) | 36‑1.csv | “Production Cost ($/100 kg)” |
| Daily quota | \(q_i\) | 36‑1.csv | “Production Quota (max per day)” |
| Activation cost | \(a_i\) | 36‑2.csv | “Activation Cost ($)” |
| Minimum batch size | \(b_i\) | 36‑3.csv | “Minimum Batch Size (100 kg units)” |

Let \(D=22\) be the number of production days available.

### Objective
\[
\max \;\sum_{i\in I}\bigl[(p_i-c_i)\,x_i - a_i\,y_i\bigr]
\]

### Constraints
1. **Demand limits**  
   \[
   x_i \le d_i \qquad \forall i\in I
   \]

2. **Shared production‑day capacity**  
   \[
   \sum_{i\in I}\frac{x_i}{q_i}\;\le\; D
   \]

3. **Activation‑dependent production**  
   \[
   x_i \;\le\; d_i\,y_i \qquad \forall i\in I
   \]

4. **Minimum batch size**  
   \[
   x_i \;\ge\; b_i\,y_i \qquad \forall i\in I
   \]

5. **Variable domains**  
   \[
   x_i \in \mathbb Z_{\ge 0}, \qquad y_i \in \{0,1\}\quad \forall i\in I
   \]

---

## Data Mapping

| Table ID | Column | Mapping |
|----------|--------|---------|
| `file_0_view_0` | `Maximum Demand (100 kg units)` | \(d_i\) |
| `file_0_view_0` | `Selling Price ($/100 kg)` | \(p_i\) |
| `file_0_view_0` | `Production Cost ($/100 kg)` | \(c_i\) |
| `file_0_view_0` | `Production Quota (max per day)` | \(q_i\) |
| `file_1_view_0` | `Activation Cost ($)` | \(a_i\) |
| `file_2_view_0` | `Minimum Batch Size (100 kg units)` | \(b_i\) |

All parameters are taken directly from the corresponding columns; no additional processing or aggregation is performed. The model uses the full set of 80 products as defined in the CSV files.