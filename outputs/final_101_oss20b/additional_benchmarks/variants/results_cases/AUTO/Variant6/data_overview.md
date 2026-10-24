{
  "query": "A retailer is designing a promotional bundle composed of one option from each product family. The value, shelf weight, and budget usage of each option are given in option_catalog.csv. The overall shelf-weight and budget limits are given in resource_limits.csv.\n\nFormulate a binary multi-choice knapsack model. For each family f and option o, define x_fo as a binary variable equal to 1 if option o is selected for family f and 0 otherwise. The objective is to maximize total bundle value. The model should select exactly one option from each family, satisfy the total weight and budget limits, and impose binary restrictions on all selection variables.",
  "relationships": [],
  "tables": [
    {
      "columns": [
        "Family",
        "Option",
        "Value",
        "Weight",
        "BudgetUse"
      ],
      "file_name": "option_catalog.csv",
      "records": [
        {
          "data_row_number": 0,
          "values": {
            "BudgetUse": "10",
            "Family": "F1",
            "Option": "O1",
            "Value": "28",
            "Weight": "8"
          }
        },
        {
          "data_row_number": 1,
          "values": {
            "BudgetUse": "13",
            "Family": "F1",
            "Option": "O2",
            "Value": "34",
            "Weight": "11"
          }
        },
        {
          "data_row_number": 2,
          "values": {
            "BudgetUse": "12",
            "Family": "F1",
            "Option": "O3",
            "Value": "30",
            "Weight": "9"
          }
        },
        {
          "data_row_number": 3,
          "values": {
            "BudgetUse": "9",
            "Family": "F1",
            "Option": "O4",
            "Value": "24",
            "Weight": "7"
          }
        },
        {
          "data_row_number": 4,
          "values": {
            "BudgetUse": "9",
            "Family": "F2",
            "Option": "O1",
            "Value": "25",
            "Weight": "7"
          }
        },
        {
          "data_row_number": 5,
          "values": {
            "BudgetUse": "12",
            "Family": "F2",
            "Option": "O2",
            "Value": "31",
            "Weight": "10"
          }
        },
        {
          "data_row_number": 6,
          "values": {
            "BudgetUse": "14",
            "Family": "F2",
            "Option": "O3",
            "Value": "36",
            "Weight": "12"
          }
        },
        {
          "data_row_number": 7,
          "values": {
            "BudgetUse": "11",
            "Family": "F2",
            "Option": "O4",
            "Value": "29",
            "Weight": "9"
          }
        },
        {
          "data_row_number": 8,
          "values": {
            "BudgetUse": "12",
            "Family": "F3",
            "Option": "O1",
            "Value": "33",
            "Weight": "10"
          }
        },
        {
          "data_row_number": 9,
          "values": {
            "BudgetUse": "10",
            "Family": "F3",
            "Option": "O2",
            "Value": "27",
            "Weight": "8"
          }
        },
        {
          "data_row_number": 10,
          "values": {
            "BudgetUse": "15",
            "Family": "F3",
            "Option": "O3",
            "Value": "38",
            "Weight": "13"
          }
        },
        {
          "data_row_number": 11,
          "values": {
            "BudgetUse": "11",
            "Family": "F3",
            "Option": "O4",
            "Value": "30",
            "Weight": "9"
          }
        },
        {
          "data_row_number": 12,
          "values": {
            "BudgetUse": "8",
            "Family": "F4",
            "Option": "O1",
            "Value": "26",
            "Weight": "6"
          }
        },
        {
          "data_row_number": 13,
          "values": {
            "BudgetUse": "13",
            "Family": "F4",
            "Option": "O2",
            "Value": "35",
            "Weight": "11"
          }
        },
        {
          "data_row_number": 14,
          "values": {
            "BudgetUse": "12",
            "Family": "F4",
            "Option": "O3",
            "Value": "32",
            "Weight": "10"
          }
        },
        {
          "data_row_number": 15,
          "values": {
            "BudgetUse": "10",
            "Family": "F4",
            "Option": "O4",
            "Value": "28",
            "Weight": "8"
          }
        },
        {
          "data_row_number": 16,
          "values": {
            "BudgetUse": "11",
            "Family": "F5",
            "Option": "O1",
            "Value": "30",
            "Weight": "9"
          }
        },
        {
          "data_row_number": 17,
          "values": {
            "BudgetUse": "14",
            "Family": "F5",
            "Option": "O2",
            "Value": "37",
            "Weight": "12"
          }
        },
        {
          "data_row_number": 18,
          "values": {
            "BudgetUse": "10",
            "Family": "F5",
            "Option": "O3",
            "Value": "29",
            "Weight": "8"
          }
        },
        {
          "data_row_number": 19,
          "values": {
            "BudgetUse": "13",
            "Family": "F5",
            "Option": "O4",
            "Value": "34",
            "Weight": "11"
          }
        },
        {
          "data_row_number": 20,
          "values": {
            "BudgetUse": "9",
            "Family": "F6",
            "Option": "O1",
            "Value": "24",
            "Weight": "7"
          }
        },
        {
          "data_row_number": 21,
          "values": {
            "BudgetUse": "12",
            "Family": "F6",
            "Option": "O2",
            "Value": "32",
            "Weight": "10"
          }
        },
        {
          "data_row_number": 22,
          "values": {
            "BudgetUse": "14",
            "Family": "F6",
            "Option": "O3",
            "Value": "36",
            "Weight": "12"
          }
        },
        {
          "data_row_number": 23,
          "values": {
            "BudgetUse": "11",
            "Family": "F6",
            "Option": "O4",
            "Value": "31",
            "Weight": "9"
          }
        }
      ],
      "table_id": "file_0_view_0"
    },
    {
      "columns": [
        "Resource",
        "Limit"
      ],
      "file_name": "resource_limits.csv",
      "records": [
        {
          "data_row_number": 0,
          "values": {
            "Limit": "55",
            "Resource": "Weight"
          }
        },
        {
          "data_row_number": 1,
          "values": {
            "Limit": "70",
            "Resource": "BudgetUse"
          }
        }
      ],
      "table_id": "file_1_view_0"
    }
  ]
}
