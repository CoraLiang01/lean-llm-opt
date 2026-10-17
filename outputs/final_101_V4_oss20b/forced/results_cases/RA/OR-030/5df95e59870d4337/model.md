**Mathematical Model**

Let  

- \(I\) be the set of all records returned from the filtered view of *BigMartSales.csv* (each record corresponds to a distinct FDK57 car model).  
- For each \(i\in I\) define parameters  

\[
\begin{aligned}
\text{rev}_i &\;=\; \text{Revenue of model }i,\\
\text{dem}_i &\;=\; \text{Demand of model }i,\\
\text{inv}_i &\;=\; \text{Initial Inventory of model }i.
\end{aligned}
\]

Decision variables  

\[
x_i \;\in\; \mathbb{Z}_{\ge 0}\qquad \forall i\in I
\]

represent the quantity of model \(i\) that will be sold.

---

**Objective**

\[
\max \;\sum_{i\in I} \text{rev}_i \, x_i
\]

---

**Constraints**

\[
\begin{aligned}
&0 \;\le\; x_i \;\le\; \text{dem}_i &&\forall i\in I \quad (\text{cannot sell more than demand})\\[4pt]
&0 \;\le\; x_i \;\le\; \text{inv}_i &&\forall i\in I \quad (\text{cannot sell more than inventory})\\[4pt]
&x_i \;\in\; \mathbb{Z}_{\ge 0} &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Table ID | Column Name |
|-----------|----------|-------------|
| \(\text{rev}_i\) | `file_0_view_0` | `Revenue` |
| \(\text{dem}_i\) | `file_0_view_0` | `Demand` |
| \(\text{inv}_i\) | `file_0_view_0` | `Initial Inventory` |

No mapping is required for the decision variables \(x_i\).