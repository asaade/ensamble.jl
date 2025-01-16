# User Guide: Constraints in `constraints.csv`

The `constraints.csv` file defines the rules for assembling your tests, including item counts, score limits, and inclusion/exclusion requirements. Each constraint is specified in a single row with these columns:

| Column            | Description                                                                                                  |
|-------------------|--------------------------------------------------------------------------------------------------------------|
| **CONSTRAINT_ID** | Unique identifier for each constraint (any string, but must be unique).                                      |
| **TYPE**          | The constraint type (e.g., `Number`, `Sum`, `Enemies`, `AllOrNone`, `Include`, `Exclude`, `MaxUse`, `Test`). |
| **CONDITION**     | Optional expression specifying which items the constraint applies to (e.g., `LEVEL == 3`).                   |
| **LB**            | Lower bound (leave blank if not used).                                                                        |
| **UB**            | Upper bound (leave blank if not used).                                                                        |
| **ONOFF**         | Toggles the constraint (`ON`, `OFF`, or empty for ON).                                                       |

> **Tip**: Use a spreadsheet (Excel, Google Sheets) for easy editing and clarity.

---

## Example File

| ID | CONSTRAINT_ID | TYPE      | CONDITION                  | LB   | UB   | ONOFF |
|----|---------------|-----------|----------------------------|------|------|-------|
| 1  | C1            | Test      |                            | 80   | 80   | ON    |
| 2  | C2            | Score     | AREA == 1                  | 18   | 22   | ON    |
| 3  | C3            | Score     | AREA == 2                  | 18   | 22   | ON    |
| 4  | C4            | Score     | AREA == 3                  | 18   | 22   | ON    |
| 5  | C5            | Score     | AREA == 4                  | 18   | 22   | ON    |
| 6  | C6            | Sum       | Words                      | 2800 | 3400 | ON    |
| 7  | C7            | Enemies   | Enemies                    |      |      | ON    |
| 8  | C8            | AllOrNone | Friends                    |      |      | ON    |
| 9  | C9            | MaxUse    |                            | 0    | 2    | ON    |
| 10 | C10           | Overlap   |                            | 10   | 10   | OFF   |
| 11 | C11           | Exclude   | CORR <= 0.12               |      |      | OFF   |
| 12 | C12           | AllOrNone | ID IN [ITEM0001,ITEM0002]  |      |      | ON    |
| 13 | C13           | Enemies   | ID IN [ITEM0036,ITEM00039] |      |      | ON    |

---

## Column Details

### 1. CONSTRAINT_ID
- Unique label for reference (e.g., `C1`, `C2`), used for debugging or logging.

### 2. TYPE
Common types include:

- **Number**: Enforces the number of selected items that satisfy a condition (e.g., 10 items must have `LEVEL == 3`).
- **Sum**: Constrains the sum of an attribute (e.g., `WORDS`) for items meeting a condition.
- **Enemies**: Items in a set cannot appear together in a form.
- **AllOrNone**: Either include all items that satisfy the condition or exclude them all.
- **Include**: Always include items that satisfy the condition.
- **Exclude**: Always exclude items that satisfy the condition.
- **MaxUse**: Limits how many times any item can appear across multiple forms.
- **Overlap**: Enforces a minimum or exact number of items shared among forms (cannot be used with anchors/shadow tests).
- **Test** / **Score**: Used for total test items or sub-scores (e.g., `Test` length must be exactly 80).

### 3. CONDITION
- Determines which items a constraint applies to.
- **Syntax**:
  - Use relational operators (`==`, `!=`, `>`, `<`, `>=`, `<=`).
  - Combine criteria with logical AND (`&&`) or OR (`||`).
  - Use `IN` for sets (e.g., `AREA IN [1, 2]` or `ID NOT IN [ITEM001, ITEM002]`).
- **Examples**:
  - `LEVEL == 3`: Applies to items where the value in column `LEVEL` is exactly 3.
  - `ID IN [A1, A2, A3]`: Applies only to items named A1, A2, or A3. The syntax
   also accepts the NOT keyword as negation. as in `ID NOT IN [A1, A2, A3]`
  - `DIFFICULTY >= 0.4 && AREA == Math`: Applies to items with a difficulty ≥ 0.4 in the Math area.
  - `WORDS, AREA == MATH`: Limits the WORDS in items from AREA == MATH.
  - `LEVEL != 2 || PTBIS > 0.3`: Applies to items that do not have level 2 OR have a point-biserial > 0.3.

If blank, the constraint applies to **all** items.

### 4. LB (Lower Bound) and UB (Upper Bound)
- Set the range for a constraint. For example:
  - **Number**: Count the number of items that from those that match the condition.
  - **Sum**: Sums the value for items matching the condition (e.g. The sum of
    `WORDS` must be 500–600 those items).
  - **Test**: Set total items in a test (e.g., 70-80 or 80-80). This is the only constraint
    that MUST be included in all cases.
- Leave blank if you do not need upper or lower limits.

### 5. ONOFF
- **ON** → activate the constraint.
- **OFF** → ignore the constraint without removing it.

---

## Example Constraints

1. **Fixed Number of Items**
   ```csv
   C1, Number, LEVEL == 3, 12, 12, ON
   ```
   Exactly 12 items where the value in column `LEVEL` is 3.

2. **Sum of an Attribute**
   ```csv
   C2, Sum, "WORDS, DIFFICULTY >= 0.4", 300, 500, ON
   ```
   For items with `DIFFICULTY >= 0.4`, the sum of `WORDS` is between 300
   and 500. In this case, the quotation marks are needed to avoid the comma between the
   column name `WORDS` and the condition breaking the constraint.

3. **Include/Exclude**
   ```csv
   C3, Exclude, PTBIS < 0.15, ON
   ```
   Exclude all items with a point-biserial < 0.15.

4. **Enemies**
   ```csv
   C4, Enemies, ID IN [A1, A2], ON
   ```
   Only one of the items with the name `A1` or `A2` can appear in a form.

5. **AllOrNone**
   ```csv
   C5, AllOrNone, ID IN [B1, B2], ON
   ```
   Either include both items with ID of B1 and B2, or exclude both.

6. **MaxUse**
   ```csv
   C6, MaxUse,, 0, 1, ON
   ```
   Each item can appear at most once across all forms. Note that the third
   column was left blank, so the limit applies to all itemss.

7. **Overlap**
   ```csv
   C7, Overlap,, 10, 10, ON
   ```
   Forms share exactly 10 items (not compatible with anchors/shadow tests).

---

## Handling Conflicts

Constraints may conflict if they impose contradictory rules (e.g., an item is forced both in and out).
- **Detection**: Some solvers will flag infeasible combinations.
- **Resolution**: Adjust or disable conflicting constraints to detect faulty or
  infeasible constraints.

### Common Conflicts

- **AllOrNone vs Enemies**: One requires items together, the other forbids them together.
- **Include vs Exclude**: Same item or set is both included and excluded.
- **Anchor Items**: May conflict with overlap or all-or-none rules if not carefully designed.

---

## Best Practices

1. **Test in Phases**: Toggle constraints (ON/OFF) in small groups to isolate potential conflicts.
2. **Use Spreadsheet Tools**: Easily filter by `TYPE` or `CONSTRAINT_ID` to ensure clarity.
3. **Unique IDs**: Keep each `CONSTRAINT_ID` clear and descriptive (e.g., `C_LEVEL3_COUNT`, `C_MATH_WORDS_SUM`).

---

## Summary

`constraints.csv` is where you define how many items to select, which to exclude, which must appear together or separately, and how item attributes should sum up. By combining logical conditions (`&&`, `||`, `IN`) with clear bounds (`LB`, `UB`), you can precisely control test form composition. Regularly check for conflicting rules to ensure a successful, error-free assembly.

