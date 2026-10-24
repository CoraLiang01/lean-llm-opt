## Mathematical Model

**Indices**  
\(I=\{1,\dots ,80\}\) – product indices.  
\(T=22\) – available production days.

**Parameters** (all taken from the CSV files)

| Parameter | Source table | Source column | Meaning |
|-----------|--------------|---------------|---------|
| \(D_i\) | `file_0_view_0` | `A{i}` in row 0 | Maximum demand (100 kg units) |
| \(P_i\) | `file_0_view_0` | `A{i}` in row 1 | Selling price ($/100 kg) |
| \(C_i\) | `file_0_view_0` | `A{i}` in row 2 | Production cost ($/100 kg) |
| \(Q_i\) | `file_0_view_0` | `A{i}` in row 3 | Production quota (max per day) |
| \(A_i\) | `file_1_view_0` | `A{i}` in row 0 | Activation cost ($) |
| \(B_i\) | `file_2_view_0` | `A{i}` in row 0 | Minimum batch size (100 kg units) |

All parameters are non‑negative integers.

**Decision variables**

\[
\begin{aligned}
x_i &\in \mathbb{Z}_{+} &&\text{units of product }i\text{ produced (100 kg)}\\
y_i &\in \{0,1\} &&\text{1 if line for product }i\text{ is activated}
\end{aligned}
\]

**Objective**

\[
\max \;\sum_{i\in I}\bigl[(P_i-C_i)\,x_i - A_i\,y_i\bigr]
\]

**Constraints**

1. **Demand limits**  
   \[
   x_i \;\le\; D_i \qquad \forall i\in I
   \]

2. **Daily quota over the month**  
   \[
   x_i \;\le\; T\,Q_i \qquad \forall i\in I
   \]

3. **Activation linkage**  
   \[
   x_i \;\le\; T\,Q_i\,y_i \qquad \forall i\in I
   \]

4. **Minimum batch size**  
   \[
   x_i \;\ge\; B_i\,y_i \qquad \forall i\in I
   \]

5. **Total production‑day capacity**  
   \[
   \sum_{i\in I}\frac{x_i}{Q_i}\;\le\; T
   \]

6. **Variable domains**  
   \[
   x_i \in \mathbb{Z}_{+},\; y_i \in \{0,1\}\qquad \forall i\in I
   \]

---

### Data Mapping

| Table ID | Column | Parameter | Row |
|----------|--------|-----------|-----|
| `file_0_view_0` | `A1`–`A80` | \(D_i\) | 0 |
| `file_0_view_0` | `A1`–`A80` | \(P_i\) | 1 |
| `file_0_view_0` | `A1`–`A80` | \(C_i\) | 2 |
| `file_0_view_0` | `A1`–`A80` | \(Q_i\) | 3 |
| `file_1_view_0` | `A1`–`A80` | \(A_i\) | 0 |
| `file_2_view_0` | `A1`–`A80` | \(B_i\) | 0 |

All constraints and the objective use only these parameters and the defined decision variables. No additional data or assumptions are introduced.