Let $x_i$ denote the number of units of Organ product $i$ to be fulfilled, where $i$ indexes the following Sub Categories:

- Organic Fruits
- Organic Staples
- Organic Vegetables

The parameters for each product $i$ are:

\[
\begin{array}{lccc}
\text{Sub Category} & \text{Revenue}_i & \text{Demand}_i & \text{Initial Inventory}_i \\
\hline
\text{Organic Fruits} & 60.8 & 678{,}906 & 5{,}034{,}020.0 \\
\text{Organic Staples} & 918.45 & 749{,}927 & 5{,}589{,}290.0 \\
\text{Organic Vegetables} & 77.52 & 699{,}808 & 5{,}202{,}710.0 \\
\end{array}
\]

The mathematical model is:

\[
\textbf{Objective:} \quad \max \; 60.8\,x_1 + 918.45\,x_2 + 77.52\,x_3
\]

\[
\textbf{Subject to:}
\]
\[
\begin{align*}
& 0 \leq x_1 \leq \min\{678{,}906,\; 5{,}034{,}020.0\} \\
& 0 \leq x_2 \leq \min\{749{,}927,\; 5{,}589{,}290.0\} \\
& 0 \leq x_3 \leq \min\{699{,}808,\; 5{,}202{,}710.0\} \\
& x_1,\,x_2,\,x_3 \in \mathbb{Z}_{\geq 0}
\end{align*}
\]

Where:
- $x_1$ = units of Organic Fruits fulfilled
- $x_2$ = units of Organic Staples fulfilled
- $x_3$ = units of Organic Vegetables fulfilled

All variables are nonnegative integers. For each product, the number of units fulfilled cannot exceed either the available initial inventory or the demand.