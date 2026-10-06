ABSTRACT MATHEMATICAL MODEL

Index Sets:
- $\mathcal{S}$: Set of bookshelves, indexed by $s$ (BookshelfID from file_0_view_0)
- $\mathcal{P}$: Set of books, indexed by $p$ (ProductName from file_1_view_0)

Parameters:
- $C_s$: Capacity of bookshelf $s$ (Capacity from file_0_view_0)
- $v_p$: Value of book $p$ (Value from file_1_view_0)
- $w_p$: Weight of book $p$ (Weight from file_1_view_0)

Decision Variables:
- $x_{sp} \in \mathbb{Z}_{\geq 0}$: Number of units of book $p$ placed on bookshelf $s$

Objective:
\[
\max \sum_{s \in \mathcal{S}} \sum_{p \in \mathcal{P}} v_p \cdot x_{sp}
\]

Subject to:
\[
\sum_{p \in \mathcal{P}} w_p \cdot x_{sp} \leq C_s \qquad \forall s \in \mathcal{S}
\]
\[
x_{sp} \in \mathbb{Z}_{\geq 0} \qquad \forall s \in \mathcal{S},\ p \in \mathcal{P}
\]

Data Mapping:

- $\mathcal{S}$: BookshelfID from file_0_view_0 (capacity.csv, preserve source order)
- $C_s$: Capacity from file_0_view_0 (capacity.csv, column "Capacity", keyed by BookshelfID)
- $\mathcal{P}$: ProductName from file_1_view_0 (products.csv, preserve source order)
- $v_p$: Value from file_1_view_0 (products.csv, column "Value", keyed by ProductName)
- $w_p$: Weight from file_1_view_0 (products.csv, column "Weight", keyed by ProductName)

All variables, parameters, and constraints are indexed and mapped exactly as in the retrieved data. No data is omitted or synthesized.