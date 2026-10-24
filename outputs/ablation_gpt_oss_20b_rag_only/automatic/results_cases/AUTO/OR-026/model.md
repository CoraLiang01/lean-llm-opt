**Mathematical Model**

Let  

- \(I\) be the set of all products whose *Product Name* contains the word “Fashion” (the set of product names returned by the filtered view).  

Parameters (indexed by \(i\in I\))  

\[
\begin{aligned}
\text{Revenue}_i &\;\;=\;\; \text{value in column “Revenue” for product } i,\\
\text{Demand}_i   &\;\;=\;\; \text{value in column “Demand” for product } i,\\
\text{Inv}_i      &\;\;=\;\; \text{value in column “Initial Inventory” for product } i.
\end{aligned}
\]

Decision variables  

\[
x_i \;\ge\; 0 \qquad \forall i\in I
\]

(continuous quantity of product \(i\) to sell).

**Objective**

\[
\max \;\; \sum_{i\in I} \text{Revenue}_i \; x_i
\]

**Constraints**

\[
\begin{aligned}
x_i &\;\le\; \text{Demand}_i &&\forall i\in I \quad\text{(cannot sell more than demand)}\\[4pt]
x_i &\;\le\; \text{Inv}_i   &&\forall i\in I \quad\text{(cannot sell more than inventory)}\\[4pt]
x_i &\;\ge\; 0              &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter / Variable | Table ID          | Column Name          |
|----------------------|-------------------|----------------------|
| \(\text{Revenue}_i\) | `file_0_view_0`   | `Revenue`            |
| \(\text{Demand}_i\)  | `file_0_view_0`   | `Demand`             |
| \(\text{Inv}_i\)     | `file_0_view_0`   | `Initial Inventory`  |
| \(x_i\)              | –                 | –                    |

*The index set \(I\) is implicitly defined by the distinct values in the “Product Name” column of the same view.*