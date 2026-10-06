##### Sets and Indices

Let $i$ index the set of all products classified as ‘Organ’:

\[
\text{Organ} = \{
\text{Organic Fruits},\ 
\text{Organic Staples},\ 
\text{Organic Vegetables}
\}
\]

##### Parameters (from source, in order)

\[
\begin{array}{llll}
\text{Product Name} & \text{Revenue}~(A_i) & \text{Demand}~(d_i) & \text{Initial Inventory}~(I_i) \\
\hline
\text{Organic Fruits} & 60.8 & 678{,}906 & 5{,}034{,}020 \\
\text{Organic Staples} & 918.45 & 749{,}927 & 5{,}589{,}290 \\
\text{Organic Vegetables} & 77.52 & 699{,}808 & 5{,}202{,}710 \\
\end{array}
\]

##### Decision Variables

\[
x_i = \text{number of units of Organ product } i \text{ to fulfill},\quad x_i \in \mathbb{Z}_+,~\forall i
\]

##### Objective Function

\[
\max \sum_{i \in \text{Organ}} A_i \cdot x_i
\]

##### Constraints

1. Inventory constraints:
\[
x_i \leq I_i,\quad \forall i \in \text{Organ}
\]

2. Demand constraints:
\[
x_i \leq d_i,\quad \forall i \in \text{Organ}
\]

3. Non-negativity and integrality:
\[
x_i \in \mathbb{Z}_+,\quad \forall i \in \text{Organ}
\]

##### Retrieved Information

{
  "Organ": [
    {
      "Product Name": "Organic Fruits",
      "Revenue": 60.8,
      "Demand": 678906,
      "Initial Inventory": 5034020.0
    },
    {
      "Product Name": "Organic Staples",
      "Revenue": 918.45,
      "Demand": 749927,
      "Initial Inventory": 5589290.0
    },
    {
      "Product Name": "Organic Vegetables",
      "Revenue": 77.52,
      "Demand": 699808,
      "Initial Inventory": 5202710.0
    }
  ]
}