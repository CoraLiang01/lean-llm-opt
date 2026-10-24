**Mathematical Model**

Sets  
\[
I \;=\; \{\text{products}\}
\]

Parameters  
\[
\begin{aligned}
\text{Revenue}_i &\quad\text{(revenue per unit of product }i)\\
\text{Demand}_i  &\quad\text{(maximum units demanded for product }i)\\
\text{InitialInv}_i &\quad\text{(initial inventory of product }i)
\end{aligned}
\]

Decision Variables  
\[
x_i \;\in\; \mathbb{Z}_{\ge 0}\quad\forall i\in I
\]

Objective  
\[
\max \;\sum_{i\in I}\text{Revenue}_i\,x_i
\]

Constraints  
\[
\begin{aligned}
x_i &\;\le\;\text{Demand}_i &&\forall i\in I\\
x_i &\;\le\;\text{InitialInv}_i &&\forall i\in I\\
x_i &\;\ge\;0 &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source File | Column Name |
|--------|-------------|-------------|
| \(I\) | `Test_Dataset/Large-scale-or/NRM_testing/NRM3/OnlineSalesDataset.csv` | `Product Name` |
| \(\text{Revenue}_i\) | same file | `Revenue` |
| \(\text{Demand}_i\) | same file | `Demand` |
| \(\text{InitialInv}_i\) | same file | `Initial Inventory` |
| \(x_i\) | decision variable | – |

All parameters are taken directly from the specified CSV file; no additional data are introduced. The model respects the inventory and demand limits and maximizes total revenue.