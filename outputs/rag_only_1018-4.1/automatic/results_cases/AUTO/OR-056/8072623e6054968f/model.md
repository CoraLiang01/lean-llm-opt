Let:
- \( x_{ij} \) = number of vessels of type \( j \) (ProductName) to be placed in display area \( i \) (DisplayID), for all \( i \in \{1,2,\ldots,14\} \), \( j \in \) {Speedboat, Fishing Boat, Catamaran, Yacht, Sailboat, Kayak, Canoe, Houseboat, Pontoon, Jet Ski, Rowboat, Hovercraft, Cabin Cruiser, Wakeboard Boat, Dinghy, Trawler, Paddle Boat, Submarine, RIB, Skiff}.

All \( x_{ij} \) are nonnegative integers.

Parameters (from products.csv, in source order):

| j (ProductName)      | Value  | Weight |
|----------------------|--------|--------|
| Speedboat            | 29664  | 18     |
| Fishing Boat         | 31778  | 36     |
| Catamaran            | 73501  | 25     |
| Yacht                | 78255  | 16     |
| Sailboat             | 93606  | 97     |
| Kayak                | 46983  | 35     |
| Canoe                | 95026  | 32     |
| Houseboat            | 57685  | 100    |
| Pontoon              | 60323  | 43     |
| Jet Ski              | 91224  | 15     |
| Rowboat              | 44003  | 95     |
| Hovercraft           | 75998  | 57     |
| Cabin Cruiser        | 84525  | 13     |
| Wakeboard Boat       | 66207  | 44     |
| Dinghy               | 65002  | 64     |
| Trawler              | 33132  | 88     |
| Paddle Boat          | 69239  | 42     |
| Submarine            | 66948  | 46     |
| RIB                  | 88240  | 24     |
| Skiff                | 48858  | 93     |

Parameters (from capacity.csv, in source order):

| i (DisplayID) | Capacity |
|---------------|----------|
| 1             | 457      |
| 2             | 604      |
| 3             | 751      |
| 4             | 468      |
| 5             | 343      |
| 6             | 408      |
| 7             | 741      |
| 8             | 914      |
| 9             | 682      |
| 10            | 409      |
| 11            | 342      |
| 12            | 903      |
| 13            | 680      |
| 14            | 886      |

Model:

Maximize total value:
\[
\text{Maximize} \quad Z = \sum_{i=1}^{14} \sum_{j=1}^{20} \text{Value}_j \cdot x_{ij}
\]
where \(\text{Value}_j\) is as listed above for each ProductName.

Subject to, for each display area \( i \) (DisplayID as above):

\[
\sum_{j=1}^{20} \text{Weight}_j \cdot x_{ij} \leq \text{Capacity}_i
\]
where \(\text{Weight}_j\) is as listed above for each ProductName, and \(\text{Capacity}_i\) is as listed above for each DisplayID.

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \text{for all } i=1,\ldots,14,\ j=1,\ldots,20
\]

All coefficients and identifiers are taken directly from the supplied CSVs, preserving their order.

Summary:
- Decision variables: \( x_{ij} \) = number of vessels of type \( j \) in display area \( i \), integer, \(\geq 0\).
- Objective: maximize total value of all vessels displayed.
- Constraints: for each display area, total vessel size (sum over all types) does not exceed its capacity.
- All data and indices are as given in the CSVs.