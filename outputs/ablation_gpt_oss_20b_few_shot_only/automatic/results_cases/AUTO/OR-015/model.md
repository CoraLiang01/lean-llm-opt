**Mathematical Model**

| Symbol | Meaning |
|--------|---------|
| \(I\) | Set of Aalop products (product names that contain the string “Aalop”) |
| \(r_i\) | Revenue per unit of product \(i\) |
| \(d_i\) | Demand for product \(i\) over the sales horizon |
| \(I_i\) | Initial inventory of product \(i\) |
| \(x_i\) | Integer number of units of product \(i\) the restaurant intends to sell |

**Objective**

\[
\max \;\sum_{i\in I} r_i\,x_i
\]

**Constraints**

\[
\begin{aligned}
&x_i \;\le\; d_i &&\forall i\in I \quad &&\text{(cannot sell more than demand)}\\[4pt]
&x_i \;\le\; I_i &&\forall i\in I \quad &&\text{(cannot sell more than initial inventory)}\\[4pt]
&x_i \;\in\; \mathbb{Z}_{+} &&\forall i\in I \quad &&\text{(positive integer sales)}
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source File | Source Column | Mapping Rule |
|-----------|-------------|---------------|--------------|
| \(r_i\) | `Test_Dataset/Large-scale-or/NRM_testing/NRM6/RestaurantSalesreport.csv` | `Revenue` | For each product name containing “Aalop”, assign the corresponding `Revenue` value to \(r_i\). |
| \(d_i\) | Same file | `Demand` | For each product name containing “Aalop”, assign the corresponding `Demand` value to \(d_i\). |
| \(I_i\) | Same file | `Initial Inventory` | For each product name containing “Aalop”, assign the corresponding `Initial Inventory` value to \(I_i\). |
| \(I\) | Same file | `Product Name` | \(I\) is the set of all product names that contain the substring “Aalop”. |

*No other data values or record counts are required for the model.*