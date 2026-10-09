Below is the complete retrieval of all data from the provided context, preserving all facility IDs, customer IDs, fixed costs, demand, and the full transportation cost matrix. All identifiers and values are retained as in the source, with explicit mapping of facilities (suppliers) and customers (stores).

---

### 1. Fixed Costs (from `fixed_cost.csv`)
| Facility (Supplier) | Fixed Cost | Source Row (archive_revision_number) |
|---------------------|------------|--------------------------------------|
| MOUNT AYR           | 96.58      | 4                                    |
| WAUKEE              | 94.06      | 6                                    |
| WAVERLY             | 94.37      | 1                                    |
| PELLA               | 82.88      | 4                                    |
| DES MOINES          | 94.96      | 3                                    |

---

### 2. Demand (from `demand.csv`)
| Customer (Store) | Demand | Source Row (archive_revision_number) |
|------------------|--------|--------------------------------------|
| Customer_1       | 2397   | 1                                    |
| Customer_2       | 1889   | 2                                    |
| Customer_3       | 2518   | 2                                    |
| Customer_4       | 3218   | 5                                    |
| Customer_5       | 1813   | 5                                    |

---

### 3. Transportation Costs (from `transportation_costs.csv`)
#### Each row is a supplier (facility), each column is a store (customer). Values are per-unit transportation costs.

#### a. MOUNT AYR (archive_revision_number: 6)
| To Customer      | Cost   | Source Row |
|------------------|--------|------------|
| CLARINDA         | 694.68 | 6          |
| FORT MADISON     | 17.48  | 6          |
| SIOUX CITY       | 20.07  | 6          |
| TOLEDO           | 199.02 | 6          |
| BANCROFT         | 1685.53| 6          |

#### b. WAUKEE (archive_revision_number: 1)
| To Customer      | Cost   | Source Row |
|------------------|--------|------------|
| CLARINDA         | 15.13  | 1          |
| FORT MADISON     | 1.50   | 1          |
| SIOUX CITY       | 1.43   | 1          |
| TOLEDO           | 27.88  | 1          |
| BANCROFT         | 90.69  | 1          |

#### c. WAVERLY (archive_revision_number: 6)
| To Customer      | Cost   | Source Row |
|------------------|--------|------------|
| CLARINDA         | 2.34   | 6          |
| FORT MADISON     | 349.34 | 6          |
| SIOUX CITY       | 246.60 | 6          |
| TOLEDO           | 41.30  | 6          |
| BANCROFT         | 78.73  | 6          |

#### d. PELLA (archive_revision_number: 1)
| To Customer      | Cost    | Source Row |
|------------------|---------|------------|
| CLARINDA         | 1181.60 | 1          |
| FORT MADISON     | 1458.53 | 1          |
| SIOUX CITY       | 1646.36 | 1          |
| TOLEDO           | 1924.55 | 1          |
| BANCROFT         | 38.93   | 1          |

#### e. DES MOINES (archive_revision_number: 4)
| To Customer      | Cost    | Source Row |
|------------------|---------|------------|
| CLARINDA         | 1030.80 | 4          |
| FORT MADISON     | 43.48   | 4          |
| SIOUX CITY       | 932.43  | 4          |
| TOLEDO           | 55.39   | 4          |
| BANCROFT         | 103.84  | 4          |

---

### 4. Source Orientation and Shape

- **Facilities (Suppliers):** MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES
- **Customers (Stores):** CLARINDA, FORT MADISON, SIOUX CITY, TOLEDO, BANCROFT
- **Fixed Costs:** One per facility, as above.
- **Demand:** Provided for Customer_1 to Customer_5 (mapping to stores is not explicit in the data provided; further mapping may be required).
- **Transportation Cost Matrix:** 5 (facilities) × 5 (customers), as above.

---

**All data is preserved as in the original context, with explicit facility and customer IDs, fixed costs, demand, and transportation costs, along with their source-row positions and archive revision numbers. No data has been omitted, transposed, or inferred beyond the original context.**