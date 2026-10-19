**Question:**  
Formulate a complete mathematical optimization model (abstract form) for the “4U” products using the data retrieved from the CSV file. The model should include index sets, parameters, decision variables, objective function, and all constraints. Provide a short Data Mapping section that identifies the source table and columns used. Do not include any explanatory text beyond the model and mapping.  

**Thought:**  
The observation returned three rows from the table `file_0_view_0` where the product name contains “4U”.  
We will index these rows as \(i = 1,2,3\).  
Parameters:  
- \(w_i\): revenue per unit (from column “Revenue”)  
- \(d_i\): demand (from column “Demand”)  
- \(c_i\): initial inventory (from column “Initial Inventory”)  

Decision variable:  
- \(x_i\): units of product \(i\) to fulfill (integer, \(\ge 0\)).  

Objective: maximize total revenue \(\sum_i w_i x_i\).  

Constraints:  
- Demand limit: \(x_i \le d_i\).  
- Inventory limit: \(x_i \le c_i\).  

No additional constraints are specified.  

**Final Answer**

##### Abstract Model

**Index Set**  
\(I = \{1,2,3\}\)

**Parameters**  
\(w_i \in \mathbb{R}_{+}\) (revenue per unit for product \(i\))  
\(d_i \in \mathbb{Z}_{+}\) (demand for product \(i\))  
\(c_i \in \mathbb{Z}_{+}\) (initial inventory for product \(i\))

**Decision Variables**  
\(x_i \in \mathbb{Z}_{+}\) (number of units of product \(i\) to fulfill)

**Objective Function**  
\[
\max \;\sum_{i \in I} w_i \, x_i
\]

**Constraints**  

1. **Demand Constraints**  
\[
x_i \;\le\; d_i , \qquad \forall i \in I
\]

2. **Inventory Constraints**  
\[
x_i \;\le\; c_i , \qquad \forall i \in I
\]

3. **Variable Domain**  
\[
x_i \in \mathbb{Z}_{+}, \qquad \forall i \in I
\]

##### Data Mapping

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(w_i\)   | `file_0_view_0` | `Revenue` |
| \(d_i\)   | `file_0_view_0` | `Demand` |
| \(c_i\)   | `file_0_view_0` | `Initial Inventory` |

*The table `file_0_view_0` contains the filtered rows where “Product Name” contains “4U”.*