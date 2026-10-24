**Mathematical Model**

Let  

* \(I\) – index set of all products whose *Product Name* contains the word “Fashion” (the 49 rows returned).  
* For each \(i\in I\)  

\[
\begin{aligned}
r_i &\;=\;\text{Revenue of product }i \\
d_i &\;=\;\text{Demand of product }i \\
s_i &\;=\;\text{Initial Inventory of product }i \\
x_i &\;=\;\text{number of units of product }i\text{ to be sold}
\end{aligned}
\]

---

### Objective
\[
\max \;\sum_{i\in I} r_i\,x_i
\]

### Constraints
1. **Demand limit**  
   \[
   x_i \;\le\; d_i \qquad \forall i\in I
   \]

2. **Inventory limit**  
   \[
   x_i \;\le\; s_i \qquad \forall i\in I
   \]

3. **Non‑negativity / integrality**  
   \[
   x_i \in \mathbb{Z}_{\ge 0}\qquad \forall i\in I
   \]

---

**Data Mapping**

| Symbol | Source Table | Source Column | Mapping Rule |
|--------|--------------|---------------|--------------|
| \(I\) | `file_0_view_0` | `Product Name` | Each distinct product name containing “Fashion” forms an index element. |
| \(r_i\) | `file_0_view_0` | `Revenue` | \(r_i\) is the numeric value in the *Revenue* column for product \(i\). |
| \(d_i\) | `file_0_view_0` | `Demand` | \(d_i\) is the numeric value in the *Demand* column for product \(i\). |
| \(s_i\) | `file_0_view_0` | `Initial Inventory` | \(s_i\) is the numeric value in the *Initial Inventory* column for product \(i\). |
| \(x_i\) | – | – | Decision variable defined in the model. |

*No other tables or columns are referenced.*