## Mathematical Model

**Index set**  
\(I=\{\text{all products in 41.csv}\}\)

**Parameters**  
\[
\begin{aligned}
L_i &\;=\;\text{Labor per unit for product }i \quad (\text{units of labor})\\
M_i &\;=\;\text{Material per unit for product }i \quad (\text{units of material})\\
P_i &\;=\;\text{Selling price per unit for product }i \quad (\$)\\
C_i &\;=\;\text{Variable cost per unit for product }i \quad (\$)\\
F   &\;=\;4500 \quad (\$ \text{ fixed operating cost per week})\\
L^{\max} &\;=\;1650 \quad (\text{weekly labor capacity})\\
M^{\max} &\;=\;1850 \quad (\text{weekly material capacity})
\end{aligned}
\]

**Decision variables**  
\[
x_i \;\ge\;0 \quad\text{(continuous production quantity of product }i\text{)}
\]

**Objective**  
\[
\max \;\sum_{i\in I}\bigl(P_i-C_i\bigr)\,x_i \;-\; F
\]

**Constraints**  

Labor constraint  
\[
\sum_{i\in I} L_i\,x_i \;\le\; L^{\max}
\]

Material constraint  
\[
\sum_{i\in I} M_i\,x_i \;\le\; M^{\max}
\]

All variables are non‑negative continuous.

---

## Data Mapping

| Parameter | Source | Table ID | Column Name |
|-----------|--------|----------|-------------|
| \(L_i\) | Labor per unit | file_0_view_0 | Labor per unit |
| \(M_i\) | Material per unit | file_0_view_0 | Material per unit |
| \(P_i\) | Selling Price | file_0_view_0 | Selling Price |
| \(C_i\) | Variable Cost | file_0_view_0 | Variable Cost |
| \(F\) | Fixed operating cost | *constant* | 4500 |
| \(L^{\max}\) | Weekly labor capacity | *constant* | 1650 |
| \(M^{\max}\) | Weekly material capacity | *constant* | 1850 |

The index set \(I\) consists of all product names returned in the 198 rows of **41.csv**. No additional filtering or aggregation is applied. All parameters are taken directly from the corresponding columns of the source table. The objective and constraints use these parameters exactly as defined.