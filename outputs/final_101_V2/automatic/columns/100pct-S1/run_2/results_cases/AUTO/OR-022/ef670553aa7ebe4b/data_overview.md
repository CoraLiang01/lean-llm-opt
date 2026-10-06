Below is all data required to formulate the described facility location and transportation model, preserving source order, exact identifiers, and values. Each facility (supplier) and customer (branch) is listed with its fixed cost, demand, and the full cost matrix (transportation cost per unit from each facility to each customer). All identifiers and values are retained as in the source.

---

### 1. Customer (Branch) Demand Data (from 'demand.csv')
| Source Row | customer_id | document_page_count | archive_revision_number | demand_units |
|------------|-------------|--------------------|------------------------|-------------|
| 1          | C1          | 8                  | 6                      | 143         |
| 2          | C2          | 2                  | 5                      | 6           |
| 3          | C3          | 4                  | 3                      | 10          |
| 4          | C4          | 6                  | 1                      | 25          |
| 5          | C5          | 4                  | 1                      | 3           |

---

### 2. Facility (Supplier) Fixed Cost Data (from 'fixed_cost.csv')
| Source Row | facility_id | archive_revision_number | document_page_count | fixed_opening_cost |
|------------|-------------|------------------------|--------------------|--------------------|
| 1          | S1          | 3                      | 8                  | 97.65              |
| 2          | S2          | 5                      | 12                 | 99.76              |
| 3          | S3          | 1                      | 4                  | 100.76             |
| 4          | S4          | 2                      | 2                  | 105.32             |
| 5          | S5          | 5                      | 8                  | 98.88              |

---

### 3. Transportation Cost Matrix (from 'transportation_costs.csv')
#### Each row is a facility (supplier), each column is a customer (branch). All costs are per unit.

#### Facility S1 (archive_batch_number: 301, archive_storage_medium: Digital)
| Source Row | facility_id | customer_id | transportation_cost_to_C1 | transportation_cost_to_C2 | transportation_cost_to_C3 | transportation_cost_to_C4 | transportation_cost_to_C5 |
|------------|-------------|-------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|
| 1          | S1          | C1–C5       | 150.74                   | 0.02                     | 49.13                    | 2080.15                  | 426.4                    |

#### Facility S2 (archive_batch_number: 303, archive_storage_medium: Hybrid)
| Source Row | facility_id | customer_id | transportation_cost_to_C1 | transportation_cost_to_C2 | transportation_cost_to_C3 | transportation_cost_to_C4 | transportation_cost_to_C5 |
|------------|-------------|-------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|
| 2          | S2          | C1–C5       | 233.05                   | 97.73                    | 49.84                    | 1982.39                  | 23.96                    |

#### Facility S3 (archive_batch_number: 303, archive_storage_medium: Digital)
| Source Row | facility_id | customer_id | transportation_cost_to_C1 | transportation_cost_to_C2 | transportation_cost_to_C3 | transportation_cost_to_C4 | transportation_cost_to_C5 |
|------------|-------------|-------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|
| 3          | S3          | C1–C5       | 55.68                    | 935.61                   | 4.03                     | 73.09                    | 525.32                   |

#### Facility S4 (archive_batch_number: 301, archive_storage_medium: Digital)
| Source Row | facility_id | customer_id | transportation_cost_to_C1 | transportation_cost_to_C2 | transportation_cost_to_C3 | transportation_cost_to_C4 | transportation_cost_to_C5 |
|------------|-------------|-------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|
| 4          | S4          | C1–C5       | 1483.82                  | 1801.08                  | 112.16                   | 816.05                   | 107.01                   |

#### Facility S5 (archive_batch_number: 301, archive_storage_medium: Digital)
| Source Row | facility_id | customer_id | transportation_cost_to_C1 | transportation_cost_to_C2 | transportation_cost_to_C3 | transportation_cost_to_C4 | transportation_cost_to_C5 |
|------------|-------------|-------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|
| 5          | S5          | C1–C5       | 1119.47                  | 884.31                   | 0.08                     | 1544.95                  | 543.67                   |

---

### 4. Capacity Data
- No explicit capacity data is present for any facility in the provided sources. Capacity for each facility is unresolved evidence (not zero).

---

### 5. Matrix Axis and Source Orientation
- Facilities (suppliers): S1, S2, S3, S4, S5 (rows)
- Customers (branches): C1, C2, C3, C4, C5 (columns)
- All cost and demand data are aligned to these identifiers and order.

---

**All data above is preserved in original source order and with exact identifiers and values. No transposition, truncation, or inference has been performed.**