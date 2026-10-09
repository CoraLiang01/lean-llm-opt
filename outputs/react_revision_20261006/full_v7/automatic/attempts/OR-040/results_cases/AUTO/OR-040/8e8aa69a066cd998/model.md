##### Mathematical Model

Let:
- $I$ = set of areas (indexed by $i$), with area names given by the "ProductName" column in products.csv.
- $x_i$ = integer variable representing the daily scale of development in area $i$.
- $v_i$ = benefit coefficient for area $i$ (from "Value" in products.csv).
- $w_i$ = development unit weight for area $i$ (from "Weight" in products.csv).
- $C$ = overall development capacity (from "Capacity" in capacity.csv).

Objective:
\[
\max \sum_{i \in I} v_i x_i
\]

Subject to:
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

##### Data Mapping

- $I$: All area names from file_1_view_0, column "ProductName"
- $v_i$: file_1_view_0, column "Value", keyed by "ProductName"
- $w_i$: file_1_view_0, column "Weight", keyed by "ProductName"
- $C$: file_0_view_0, column "Capacity"
- $x_i$: integer, nonnegative, for each $i \in I$