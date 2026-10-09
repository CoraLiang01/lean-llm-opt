**Retrieved Data for Facility Location/Inventory Replenishment Model**

---

### 1. Facility (Supplier) Data: Fixed Costs

| Source Row | facility_id | archive_batch_number | archive_revision_number | record_view_count | document_page_count | fixed_opening_cost |
|------------|-------------|---------------------|------------------------|-------------------|--------------------|--------------------|
| 6          | S1          | 305                 | 3                      | 12                | 8                  | 97.65              |
| 7          | S2          | 302                 | 5                      | 58                | 12                 | 99.76              |
| 8          | S3          | 301                 | 1                      | 12                | 4                  | 100.76             |
| 9          | S4          | 301                 | 2                      | 76                | 2                  | 105.32             |
| 10         | S5          | 305                 | 5                      | 27                | 8                  | 98.88              |

---

### 2. Customer (Branch) Demand Data

| Source Row | customer_id | archive_batch_number | archive_revision_number | record_view_count | document_page_count | demand_units |
|------------|-------------|---------------------|------------------------|-------------------|--------------------|--------------|
| 1          | C1          | 305                 | 6                      | 27                | 8                  | 143          |
| 2          | C2          | 303                 | 5                      | 76                | 2                  | 6            |
| 3          | C3          | 302                 | 3                      | 43                | 4                  | 10           |
| 4          | C4          | 301                 | 1                      | 27                | 6                  | 25           |
| 5          | C5          | 303                 | 1                      | 12                | 4                  | 3            |

---

### 3. Transportation Cost Matrix (per unit, by facility and customer)

#### Source Row 11: Facility S1, archive_batch_number 301, archive_revision_number 2

| facility_id | customer_id | transportation_cost |
|-------------|-------------|---------------------|
| S1          | C1          | 150.74              |
| S1          | C2          | 0.02                |
| S1          | C3          | 49.13               |
| S1          | C4          | 2080.15             |
| S1          | C5          | 426.4               |

#### Source Row 12: Facility S2, archive_batch_number 303, archive_revision_number 2

| facility_id | customer_id | transportation_cost |
|-------------|-------------|---------------------|
| S2          | C1          | 233.05              |
| S2          | C2          | 97.73               |
| S2          | C3          | 49.84               |
| S2          | C4          | 1982.39             |
| S2          | C5          | 23.96               |

#### Source Row 13: Facility S3, archive_batch_number 303, archive_revision_number 2

| facility_id | customer_id | transportation_cost |
|-------------|-------------|---------------------|
| S3          | C1          | 55.68               |
| S3          | C2          | 935.61              |
| S3          | C3          | 4.03                |
| S3          | C4          | 73.09               |
| S3          | C5          | 525.32              |

#### Source Row 14: Facility S4, archive_batch_number 301, archive_revision_number 4

| facility_id | customer_id | transportation_cost |
|-------------|-------------|---------------------|
| S4          | C1          | 1483.82             |
| S4          | C2          | 1801.08             |
| S4          | C3          | 112.16              |
| S4          | C4          | 816.05              |
| S4          | C5          | 107.01              |

#### Source Row 15: Facility S5, archive_batch_number 301, archive_revision_number 5

| facility_id | customer_id | transportation_cost |
|-------------|-------------|---------------------|
| S5          | C1          | 1119.47             |
| S5          | C2          | 884.31              |
| S5          | C3          | 0.08                |
| S5          | C4          | 1544.95             |
| S5          | C5          | 543.67              |

---

### 4. Capacity Data

- **No explicit capacity values are present in the provided data for any facility.**  
  (Absence of capacity is unresolved evidence, not zero.)

---

### 5. Source Row and Orientation

- **Facilities (Suppliers):** S1, S2, S3, S4, S5 (rows of the cost matrix, fixed costs, and capacities)
- **Customers (Branches):** C1, C2, C3, C4, C5 (columns of the cost matrix, demand)

---

### 6. Matrix Shape and Axis

- **Transportation Cost Matrix:** 5 facilities (rows) × 5 customers (columns)
- **Fixed Cost Vector:** 5 facilities (S1–S5)
- **Demand Vector:** 5 customers (C1–C5)
- **Capacity:** Not present (unresolved)

---

**All data is preserved in original source order and with explicit identifiers. No transposition, truncation, or inference has been performed.**