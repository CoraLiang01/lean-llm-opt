{
  "query": "A pharmacy chain needs to replenish its drug inventory. The ‘products.csv’ file provides a table of benefits associated with each drug product. There are overall stock capacity limits for pharmacy chains detailed in the ‘capacity.csv’ file.Our objective is to decide which drugs to order each day and in what quantities to maximise the overall benefit while adhering to the overall stock capacity. The decision variable x_i represents the number of units of the ith drug to be ordered each day.",
  "relationships": [],
  "tables": [
    {
      "columns": [
        "Capacity"
      ],
      "file_name": "capacity.csv",
      "records": [
        {
          "data_row_number": 0,
          "values": {
            "Capacity": "520"
          }
        }
      ],
      "table_id": "file_0_view_0"
    },
    {
      "columns": [
        "ProductName",
        "Value",
        "Weight"
      ],
      "file_name": "products.csv",
      "records": [
        {
          "data_row_number": 0,
          "values": {
            "ProductName": "NSAIDs",
            "Value": "250",
            "Weight": "913"
          }
        },
        {
          "data_row_number": 1,
          "values": {
            "ProductName": "Antirheumatic Drugs",
            "Value": "178",
            "Weight": "754"
          }
        },
        {
          "data_row_number": 2,
          "values": {
            "ProductName": "Acetic Acid Derivatives",
            "Value": "313",
            "Weight": "428"
          }
        },
        {
          "data_row_number": 3,
          "values": {
            "ProductName": "Antibiotics",
            "Value": "301",
            "Weight": "711"
          }
        },
        {
          "data_row_number": 4,
          "values": {
            "ProductName": "Antiviral Drugs",
            "Value": "425",
            "Weight": "350"
          }
        },
        {
          "data_row_number": 5,
          "values": {
            "ProductName": "Antifungal Agents",
            "Value": "260",
            "Weight": "159"
          }
        },
        {
          "data_row_number": 6,
          "values": {
            "ProductName": "Antidepressants",
            "Value": "848",
            "Weight": "353"
          }
        },
        {
          "data_row_number": 7,
          "values": {
            "ProductName": "Antipsychotics",
            "Value": "934",
            "Weight": "291"
          }
        },
        {
          "data_row_number": 8,
          "values": {
            "ProductName": "Antihistamines",
            "Value": "114",
            "Weight": "302"
          }
        },
        {
          "data_row_number": 9,
          "values": {
            "ProductName": "Corticosteroids",
            "Value": "1357",
            "Weight": "50"
          }
        },
        {
          "data_row_number": 10,
          "values": {
            "ProductName": "Beta Blockers",
            "Value": "156",
            "Weight": "250"
          }
        },
        {
          "data_row_number": 11,
          "values": {
            "ProductName": "Calcium Channel Blockers",
            "Value": "1780",
            "Weight": "178"
          }
        },
        {
          "data_row_number": 12,
          "values": {
            "ProductName": "ACE Inhibitors",
            "Value": "695",
            "Weight": "313"
          }
        },
        {
          "data_row_number": 13,
          "values": {
            "ProductName": "Angiotensin II Receptor Blockers",
            "Value": "405",
            "Weight": "378"
          }
        },
        {
          "data_row_number": 14,
          "values": {
            "ProductName": "Diuretics",
            "Value": "320",
            "Weight": "94"
          }
        },
        {
          "data_row_number": 15,
          "values": {
            "ProductName": "Statins",
            "Value": "320",
            "Weight": "97"
          }
        },
        {
          "data_row_number": 16,
          "values": {
            "ProductName": "Insulin",
            "Value": "1357",
            "Weight": "470"
          }
        },
        {
          "data_row_number": 17,
          "values": {
            "ProductName": "Anticoagulants",
            "Value": "1357",
            "Weight": "341"
          }
        },
        {
          "data_row_number": 18,
          "values": {
            "ProductName": "Antiepileptic Drugs",
            "Value": "405",
            "Weight": "121"
          }
        },
        {
          "data_row_number": 19,
          "values": {
            "ProductName": "Antiemetics",
            "Value": "998",
            "Weight": "61"
          }
        }
      ],
      "table_id": "file_1_view_0"
    }
  ]
}
