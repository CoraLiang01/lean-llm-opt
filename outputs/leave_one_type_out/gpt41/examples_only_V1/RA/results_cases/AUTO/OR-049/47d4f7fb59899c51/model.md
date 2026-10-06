Let:
- S = {1, 2, ..., 10} be the set of shelves (from ShelfID in capacity.csv)
- P = {1, 2, ..., 20} be the set of products, indexed in the order of products.csv
  (Product 1: Smartphone, Product 2: Laptop, ..., Product 20: E-Reader)
- Let Value_j and Weight_j be the value and weight of product j (from products.csv)
- Let Capacity_i be the capacity of shelf i (from capacity.csv)
- Decision variables: x_{i,j} = number of units of product j placed on shelf i, for i in S, j in P

Model:

Maximize total value:
\[
\text{Maximize} \quad Z = \sum_{i=1}^{10} \sum_{j=1}^{20} \text{Value}_j \cdot x_{i,j}
\]
where Value_j is as follows (in order from products.csv):
\[
\begin{align*}
\text{Value}_1 &= 200 \quad &\text{(Smartphone)} \\
\text{Value}_2 &= 1500 \quad &\text{(Laptop)} \\
\text{Value}_3 &= 100 \quad &\text{(Headphones)} \\
\text{Value}_4 &= 800 \quad &\text{(Camera)} \\
\text{Value}_5 &= 250 \quad &\text{(Smartwatch)} \\
\text{Value}_6 &= 600 \quad &\text{(Tablet)} \\
\text{Value}_7 &= 150 \quad &\text{(Bluetooth Speaker)} \\
\text{Value}_8 &= 80 \quad &\text{(Keyboard)} \\
\text{Value}_9 &= 50 \quad &\text{(Mouse)} \\
\text{Value}_{10} &= 300 \quad &\text{(Monitor)} \\
\text{Value}_{11} &= 400 \quad &\text{(Printer)} \\
\text{Value}_{12} &= 120 \quad &\text{(External Hard Drive)} \\
\text{Value}_{13} &= 60 \quad &\text{(Router)} \\
\text{Value}_{14} &= 40 \quad &\text{(Power Bank)} \\
\text{Value}_{15} &= 30 \quad &\text{(Memory Card)} \\
\text{Value}_{16} &= 25 \quad &\text{(USB Flash Drive)} \\
\text{Value}_{17} &= 100 \quad &\text{(Smart Home Hub)} \\
\text{Value}_{18} &= 500 \quad &\text{(Gaming Console)} \\
\text{Value}_{19} &= 90 \quad &\text{(Fitness Tracker)} \\
\text{Value}_{20} &= 180 \quad &\text{(E-Reader)} \\
\end{align*}
\]

Subject to shelf capacity constraints (for each shelf i):
\[
\sum_{j=1}^{20} \text{Weight}_j \cdot x_{i,j} \leq \text{Capacity}_i \qquad \forall i \in \{1,2,\ldots,10\}
\]
where Weight_j is as follows (in order from products.csv):
\[
\begin{align*}
\text{Weight}_1 &= 1.0 \\
\text{Weight}_2 &= 5.0 \\
\text{Weight}_3 &= 0.5 \\
\text{Weight}_4 &= 2.0 \\
\text{Weight}_5 &= 0.3 \\
\text{Weight}_6 &= 1.5 \\
\text{Weight}_7 &= 1.0 \\
\text{Weight}_8 &= 0.8 \\
\text{Weight}_9 &= 0.2 \\
\text{Weight}_{10} &= 3.0 \\
\text{Weight}_{11} &= 4.0 \\
\text{Weight}_{12} &= 0.5 \\
\text{Weight}_{13} &= 0.3 \\
\text{Weight}_{14} &= 0.4 \\
\text{Weight}_{15} &= 0.05 \\
\text{Weight}_{16} &= 0.02 \\
\text{Weight}_{17} &= 0.6 \\
\text{Weight}_{18} &= 4.0 \\
\text{Weight}_{19} &= 0.2 \\
\text{Weight}_{20} &= 0.5 \\
\end{align*}
\]

Shelf capacities (from capacity.csv, in order):
\[
\begin{align*}
\text{Capacity}_1 &= 5.0 \\
\text{Capacity}_2 &= 7.0 \\
\text{Capacity}_3 &= 6.0 \\
\text{Capacity}_4 &= 8.0 \\
\text{Capacity}_5 &= 5.5 \\
\text{Capacity}_6 &= 9.0 \\
\text{Capacity}_7 &= 6.5 \\
\text{Capacity}_8 &= 7.5 \\
\text{Capacity}_9 &= 8.2 \\
\text{Capacity}_{10} &= 5.7 \\
\end{align*}
\]

Variable domains:
\[
x_{i,j} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
\]

Summary:
- Decision variables: x_{i,j} = number of units of product j on shelf i (integer, ≥0)
- Objective: Maximize total value across all shelves and products
- Constraints: For each shelf, total weight of products ≤ shelf capacity
- All coefficients and IDs are as given in the CSVs above, in order.