Let $x_i$ denote the number of units of each Baby product $i$ to fulfill.

##### Sets and Parameters

- $I$: Set of Baby products (here, $I = \{\text{Baby Food\_255.28}\}$)
- For each $i \in I$:
  - $r_i$: Revenue per unit (from ‘Revenue’ column)
  - $d_i$: Demand (from ‘Demand’ column)
  - $s_i$: Initial Inventory (from ‘Initial Inventory’ column)

##### Decision Variables

- $x_i \in \mathbb{Z}_{\geq 0}$: Number of units of Baby product $i$ to fulfill

##### Model

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Subject to:
\[
x_i \leq d_i \qquad \forall i \in I
\]
\[
x_i \leq s_i \qquad \forall i \in I
\]
\[
x_i \geq 0,\quad x_i \in \mathbb{Z} \qquad \forall i \in I
\]

##### Numerical Formulation (with retrieved data)

Let $x_1$ denote the number of units of Baby Food\_255.28 to fulfill.

\[
\max\ 255.28\, x_1
\]
\[
x_1 \leq 3,\!066,\!513
\]
\[
x_1 \leq 22,\!749,\!210
\]
\[
x_1 \geq 0,\quad x_1 \in \mathbb{Z}
\]