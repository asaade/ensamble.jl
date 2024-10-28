## User Guide for Configuring Constraints in the `constraints.csv` File

The constraint configuration system allows users to specify a set of rules that an automated test assembly must adhere to when constructing test forms. The constraints.csv file defines various constraint types, ensuring that the assembled test aligns with blueprint requirements, such as the number of items, score limits, and inclusion or exclusion of specific items.

### File Structure

The file is expected to have the following columns:

| Column          | Description                                                                                     |
|-----------------|-------------------------------------------------------------------------------------------------|
| **CONSTRAINT_ID** | Unique identifier for each constraint. Can be any string but must be unique within the file. |
| **TYPE**        | Specifies the constraint type. Accepted values are `Number`, `Sum`, `Enemies`, `Include`, `Exclude`, `AllOrNone`, etc. |
| **CONDITION**   | Optional condition to filter items for which the constraint applies.                             |
| **LB**          | Lower bound for the constraint. If not required, leave empty.                                   |
| **UB**          | Upper bound for the constraint. If not required, leave empty.                                   |
| **ONOFF**       | Set to `OFF` to deactivate the constraint; otherwise, `ON` or empty will apply the constraint. |

**Note**: Using a spreadsheet application like Excel is recommended for editing this file, as it helps maintain the proper structure and visibility for each constraint.

### An example

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
| 12 | C12           | AllOrNone | ID in [ITEM0001,ITEM0002]  |      |      | ON    |
| 13 | C13           | Enemies   | ID in [ITEM0036,ITEM00039] |      |      | ON    |




### Column Descriptions

#### CONSTRAINT_ID
The unique identifier for each constraint. This can be any alphanumeric value but must be unique across all constraints in the file. It is useful for tracking and debugging.

#### TYPE
Specifies the type of constraint to apply. The available types are:

1. **Number**: Limits the count of items selected. Use the `CONDITION` column to specify a subset of items. For example:
   - **Example**: To ensure exactly 10 items with `LEVEL == 3` are selected:
     ```
     CONSTRAINT_ID, TYPE, CONDITION, LB, UB
     C1, Number, LEVEL == 3, 10, 10
     ```

2. **Sum**: Limits the sum of an item attribute across selected items. It can control the sum directly or conditionally (e.g., summing items that meet a specific criterion). This is useful for attributes like score weights or item difficulty.
   - **Example**: To ensure the sum of `WORDS` is between 500 and 600 for selected items:
     ```
     CONSTRAINT_ID, TYPE, CONDITION, LB, UB
     C2, Sum, WORDS, 500, 600
     ```

3. **Enemies**: Specifies that items meeting the condition should be mutually exclusive in the selection. This is useful when items are too similar or should not appear together in a form.
   - **Example**: Ensure only one of the two items (`ID` in [A1, A2]) is included:
     ```
     CONSTRAINT_ID, TYPE, CONDITION
     C3, Enemies, ID IN [A1, A2]
     ```

4. **Include**: Forces the inclusion of items that meet the specified condition.
   - **Example**: Always include items where `AREA == Math`:
     ```
     CONSTRAINT_ID, TYPE, CONDITION
     C4, Include, AREA == Math
     ```

5. **Exclude**: Ensures that items meeting the specified condition are excluded from the selection.
   - **Example**: Exclude items where `PTBIS < 0.15`:
     ```
     CONSTRAINT_ID, TYPE, CONDITION
     C5, Exclude, PTBIS < 0.15
     ```

6. **AllOrNone**: Ensures that all items meeting a condition are either included together or excluded entirely.
   - **Example**: Either include all items `ID IN [B1, B2]` or none:
     ```
     CONSTRAINT_ID, TYPE, CONDITION
     C6, AllOrNone, ID IN [B1, B2]
     ```

7. **Overlap**: Forces the forms to share a number of items. Is more compute-intensive and only works without anchors and shadow tests.
   - **Example**: Include 10 common items between each two forms:
     ```
     CONSTRAINT_ID, TYPE, CONDITION, LB, UB
     C7, Overlap,, 10, 10
     ```

7. **MaxUse**: Ensures that an items only appears at most that number of times in the forms.
   - **Example**: All items (except anchor items) should appear only in one form, at the most:
     ```
     CONSTRAINT_ID, TYPE, CONDITION, LB, UB
     C7, MaxUse,, 0, 1
     ```


#### CONDITION
The condition restricts the items the constraint applies to. It is optional and can be left blank if the constraint applies to all items.

Conditions are specified using *column names from the item bank file* and logical expressions (e.g., `LEVEL == 3`, `AREA == Math`). Conditions can be combined using logical operators such as `&&` (AND) and `||` (OR) or set membership operators such as `IN` (e.g., `LEVEL IN [2, 3, 4]`).

**Examples**:
- `LEVEL == 3 && AREA == Math`: Applies the constraint to items in the column 'AREA' with the name 'Math' and in the column 'LEVEL' with a value of 3.
- `DIFFICULTY >= 0.5 || PTBIS > 0.2`: Applies if difficulty is at least 0.5 or if the item discrimination index (PTBIS) is greater than 0.2.

#### LB (Lower Bound) and UB (Upper Bound)
These specify the minimum and maximum values for the constraint. For example, if `TYPE` is `Number`, `LB` and `UB` set the minimum and maximum count of items. If the `TYPE` is `Sum`, `LB` and `UB` represent the minimum and maximum allowable sums for an attribute.

**Examples**:
- To select between 10 and 15 items where `LEVEL == 3`:
  ```
  CONSTRAINT_ID, TYPE, CONDITION, LB, UB
  C7, Number, LEVEL == 3, 10, 15
  ```

- To constrain the sum of `WORDS` to be exactly 700:
  ```
  CONSTRAINT_ID, TYPE, CONDITION, LB, UB
  C8, Sum, WORDS, 700, 700
  ```

#### ONOFF
This column toggles the constraint. Set to `OFF` to deactivate the constraint. If left blank or set to `ON`, the constraint will be active.

---

### Example Configurations

#### Ensuring a Fixed Item Count with Conditions
The following configuration ensures that exactly 12 items with `LEVEL == 3` are included in the selection:
```
CONSTRAINT_ID, TYPE, CONDITION, LB, UB, ONOFF
C1, Number, LEVEL == 3, 12, 12, ON
```

#### Constraining Sum of Attributes with a Condition
To ensure that the sum of `WORDS` for items where `DIFFICULTY >= 0.4` falls between 300 and 500:
```
CONSTRAINT_ID, TYPE, CONDITION, LB, UB, ONOFF
C2, Sum, WORDS, DIFFICULTY >= 0.4, 300, 500, ON
```

#### Excluding Certain Items from Selection
To exclude items with `PTBIS < 0.15`:
```
CONSTRAINT_ID, TYPE, CONDITION, ONOFF
C3, Exclude, PTBIS < 0.15, ON
```

#### Applying Enemy Constraints
To specify that only one of the items `ID == A1` or `ID == A2` can be included:
```
CONSTRAINT_ID, TYPE, CONDITION, ONOFF
C4, Enemies, ID IN [A1, A2], ON
```

#### Ensuring All-or-None Selection for Grouped Items
To include both or none of the items where `ID` is in [B1, B2]:
```
CONSTRAINT_ID, TYPE, CONDITION, ONOFF
C5, AllOrNone, ID IN [B1, B2], ON
```


Here are examples of constraint conditions for different scenarios and `TYPE` values in the `constraints.csv` file. These examples include commonly used conditions, such as item difficulty, area, and item properties like `LEVEL` or `WORD_COUNT`, which are typical in test assembly contexts.

---

### General Structure

A constraint condition is specified in the `CONDITION` column and defines criteria that items must meet for the constraint to apply to them. The condition can be left blank if it should apply to all items.

#### Syntax Guidelines
- Conditions can include relational operators: `==`, `!=`, `>`, `<`, `>=`, `<=`.
- Logical operators `&&` (AND), `||` (OR) allow combining multiple conditions.
- The `IN` operator allows for checking set membership (e.g., `LEVEL IN [2, 3, 4]`).

---

### Condition Examples by TYPE

#### 1. **Number** - Limits the count of selected items
   - Select exactly 15 items with difficulty level 3:
     ```
     CONDITION: LEVEL == 3
     LB: 15
     UB: 15
     ```

   - Include between 10 and 20 items with `AREA == Math`:
     ```
     CONDITION: AREA == Math
     LB: 10
     UB: 20
     ```

   - Select items with `DIFFICULTY >= 0.4` and `LEVEL IN [1, 2, 3]`:
     ```
     CONDITION: DIFFICULTY >= 0.4 && LEVEL IN [1, 2, 3]
     LB: 5
     UB: 10
     ```

#### 2. **Sum** - Constrains the sum of an attribute
   - Sum of `WORD_COUNT` for all items should be between 500 and 600:
     ```
     CONDITION: WORD_COUNT
     LB: 500
     UB: 600
     ```

   - Sum of `WORD_COUNT` for items in level 2 or higher should be between 700 and 800:
     ```
     CONDITION: WORD_COUNT, LEVEL >= 2
     LB: 700
     UB: 800
     ```

   - Sum of `POINTS` for items with `AREA == Science`:
     ```
     CONDITION: POINTS, AREA == Science
     LB: 50
     UB: 70
     ```

#### 3. **Enemies** - Ensures mutual exclusivity among items
   - Only one of items with `ID` in `[A1, A2, A3]` can be included:
     ```
     CONDITION: ID IN [A1, A2, A3]
     ```

   - Select only one item with difficulty above 0.6 and level 4:
     ```
     CONDITION: DIFFICULTY > 0.6 && LEVEL == 4
     ```

#### 4. **Include** - Ensures certain items are always included
   - Always include items in `[B1, B2]`:
     ```
     CONDITION: ID IN [B1, B2]
     ```

   - Always include items with difficulty above 0.5 in the Math area:
     ```
     CONDITION: DIFFICULTY > 0.5 && AREA == Math
     ```

#### 5. **Exclude** - Ensures certain items are never included
   - Exclude items with point-biserial correlation below 0.15:
     ```
     CONDITION: PTBIS < 0.15
     ```

   - Exclude items where `LEVEL == 1` and `DIFFICULTY <= 0.3`:
     ```
     CONDITION: LEVEL == 1 && DIFFICULTY <= 0.3
     ```

#### 6. **AllOrNone** - Ensures all items in a condition group are included or excluded together
   - Either include all or none of the items with `ID IN [C1, C2, C3]`:
     ```
     CONDITION: ID IN [C1, C2, C3]
     ```

   - Either include all or none of the items with `LEVEL == 2` in the Science area:
     ```
     CONDITION: LEVEL == 2 && AREA == Science
     ```

---

### Combined Conditions

You can use `&&` (AND) and `||` (OR) operators to create more complex conditions. Here are examples:

- **Selecting items based on multiple attributes**: Include items where `LEVEL == 2` and `DIFFICULTY >= 0.3`.
  ```
  CONDITION: LEVEL == 2 && DIFFICULTY >= 0.3
  ```

- **Ensuring diversity in item attributes**: Require items to have either a low difficulty (`DIFFICULTY <= 0.3`) or high discrimination (`PTBIS >= 0.4`).
  ```
  CONDITION: DIFFICULTY <= 0.3 || PTBIS >= 0.4
  ```

- **Nested conditions with sets**: Select items where `STANDARD IN [1, 2, 3]` and the level is greater than 2.
  ```
  CONDITION: STANDARD IN [1, 2, 3] && LEVEL > 2
  ```


Conditional sum constraints are a powerful tool for managing the distribution and balance of test attributes in an assembled form. These constraints ensure that the sum of a particular attribute (e.g., `WORD_COUNT`, `DIFFICULTY`, or other item properties) for items that meet a specific condition remains within specified bounds.

This approach is especially helpful in creating balanced test forms that meet specific requirements for areas like difficulty, length, topic coverage, and cognitive levels.

---

### Structure of Conditional Sum Constraints

The general structure for specifying a conditional sum constraint in the `constraints.csv` file is as follows:

| CONSTRAINT_ID | TYPE | WHAT | CONDITION                  | LB  | UB  | ONOFF |
|---------------|------|------|----------------------------|-----|-----|-------|
| C1            | Sum  | Item | `ATTRIBUTE, CONDITION_EXPR`| 100 | 200 | ON    |

- **CONSTRAINT_ID**: A unique identifier for the constraint.
- **TYPE**: Set to `Sum` to indicate a sum constraint.
- **WHAT**: Specifies the unit, typically set to `Item`.
- **CONDITION**: The core of a conditional sum constraint, defined as:
  - `ATTRIBUTE`: The attribute to be summed across selected items (e.g., `WORD_COUNT`).
  - `CONDITION_EXPR`: An optional condition expression specifying which items to include in the sum.
- **LB** and **UB**: The lower and upper bounds for the sum.
- **ONOFF**: Turns the constraint on (`ON`) or off (`OFF`).

---

### Example Breakdown

Let’s walk through several examples to clarify how these conditional sum constraints work.

---

#### 1. Basic Sum Constraint Without a Condition

This is a simple sum constraint without conditions applied, meaning it sums an attribute across all items:

| CONSTRAINT_ID | TYPE | WHAT | CONDITION | LB  | UB  | ONOFF |
|---------------|------|------|-----------|-----|-----|-------|
| C1            | Sum  | Item | `WORD_COUNT` | 500 | 600 | ON    |

- **Explanation**: This constraint requires the total `WORD_COUNT` of all selected items to be between 500 and 600.
- **Application**: Every item’s `WORD_COUNT` attribute is included in the sum, regardless of any specific attributes.

---

#### 2. Conditional Sum Based on a Specific Attribute

This example demonstrates applying a condition on a different attribute, such as `LEVEL`:

| CONSTRAINT_ID | TYPE | WHAT | CONDITION           | LB  | UB  | ONOFF |
|---------------|------|------|---------------------|-----|-----|-------|
| C2            | Sum  | Item | `WORD_COUNT, LEVEL == 2` | 300 | 400 | ON    |

- **Explanation**: Here, the sum of `WORD_COUNT` for items where `LEVEL == 2` must fall between 300 and 400.
- **Application**: Only items with `LEVEL` equal to 2 contribute their `WORD_COUNT` to the total sum.

---

#### 3. Combining Multiple Conditions with AND (`&&`)

When you need to impose a sum constraint on items that meet multiple criteria, you can combine conditions using `&&`:

| CONSTRAINT_ID | TYPE | WHAT | CONDITION                           | LB  | UB  | ONOFF |
|---------------|------|------|-------------------------------------|-----|-----|-------|
| C3            | Sum  | Item | `WORD_COUNT, AREA == Math && LEVEL >= 3` | 200 | 500 | ON    |

- **Explanation**: The total `WORD_COUNT` of items in the Math area with `LEVEL >= 3` must be between 200 and 500.
- **Application**: Only items meeting both conditions (`AREA == Math` and `LEVEL >= 3`) are included in the sum.

---

#### 4. Conditional Sum with OR (`||`) for Broader Selection

When the conditions are less restrictive, using `||` allows items that meet at least one condition to contribute to the sum:

| CONSTRAINT_ID | TYPE | WHAT | CONDITION                                      | LB  | UB  | ONOFF |
|---------------|------|------|------------------------------------------------|-----|-----|-------|
| C4            | Sum  | Item | `WORD_COUNT, LEVEL == 2 || DIFFICULTY < 0.4`   | 250 | 400 | ON    |

- **Explanation**: This constraint sets the total `WORD_COUNT` for items with `LEVEL == 2` or `DIFFICULTY < 0.4` to be between 250 and 400.
- **Application**: Any item meeting at least one of the conditions is included in the sum.

---

#### 5. Advanced Conditional Sum Constraint with Set Membership

Conditions can also specify that an attribute’s value belongs to a specific set, using `IN`:

| CONSTRAINT_ID | TYPE | WHAT | CONDITION                             | LB  | UB  | ONOFF |
|---------------|------|------|---------------------------------------|-----|-----|-------|
| C5            | Sum  | Item | `POINTS, STANDARD IN [1, 2, 3]` | 30  | 50  | ON    |

- **Explanation**: The sum of `POINTS` for items where `STANDARD` is 1, 2, or 3 should be between 30 and 50.
- **Application**: Only items with `STANDARD` equal to 1, 2, or 3 contribute to the sum.

---

### Practical Considerations for Conditional Sum Constraints

1. **Flexibility**: Conditional sums offer precise control over various item attributes, useful for complex blueprints that may require balancing multiple dimensions (e.g., difficulty, content area).

2. **Attribute Types**: The attribute being summed (`ATTRIBUTE` in `ATTRIBUTE, CONDITION_EXPR`) can be quantitative (e.g., `WORD_COUNT`) but should match the type in `LB` and `UB` to avoid mismatched bounds.

3. **Logical Operators**: Combining conditions with `&&` and `||` allows for granular control but should be applied carefully to ensure constraints do not become too restrictive or overly lenient.

4. **Bounds Settings**: `LB` and `UB` can be equal for an exact match or set to a range for flexibility, ensuring that the sum remains within acceptable limits.

5. **Turn Constraints On/Off**: Setting `ONOFF` to `OFF` allows for easy toggling without deleting constraints. This is useful during testing or when reusing templates.


In automated test assembly systems, certain constraints might conflict with one another. Conflict handling mechanisms are essential to ensure that the constraints do not contradict each other, which would make it impossible for the solver to find a solution that satisfies all requirements. In test assembly, conflicts can arise from different types of constraints, such as inclusion and exclusion rules or requirements about mutually exclusive items, and must be carefully handled to maintain consistency.

### Conflict Handling Strategy in Constraints

The conflict handling process involves identifying, managing, and resolving conflicting constraints so the test form satisfies the requirements without violating any rules. Here is an overview of the primary components and methods used to handle conflicts in constraints:

---

### 1. **Identifying Conflicting Constraints**

Constraints can conflict due to their fundamental definitions or the item selection conditions they impose. Key types of constraints that commonly cause conflicts include:

- **Mutually Exclusive (Enemy) Constraints**: `ENEMIES` constraints specify that only one or none of the items in a specified set should be included in a form. Conflicts arise if there are other constraints that require multiple items from this set to be included together.

- **Inclusion/Exclusion Conflicts**: `INCLUDE` constraints specify that specific items must always be included in a form, whereas `EXCLUDE` constraints dictate that certain items must always be excluded. Conflicts occur if an item is listed in both `INCLUDE` and `EXCLUDE` constraints for the same form.

- **All-Or-None Constraints**: `ALLORNONE` constraints require either all items in a specified set to be included in the form or none at all. If other constraints conflict by requiring only some of these items to be included or excluded, this can lead to a conflict.

- **Conditional Constraints**: Conditions based on specific item attributes (e.g., difficulty level or content area) can also create conflicts if they overlap or contradict each other in ways that make it impossible for the system to satisfy both simultaneously.

---

### 2. **Mechanism for Conflict Detection**

The system typically uses a conflict detection mechanism that iteratively checks each pair or set of constraints to see if they can be satisfied simultaneously. This mechanism may involve:

- **Conflict Matrices**: Represent relationships among constraints to efficiently identify any conflicting pairs or groups.

- **Conflict Rules**: Apply predefined logic (e.g., only one item from an `ENEMIES` group can be included) to detect situations where constraints cannot be satisfied together.

The detection system flags constraints that, when applied together, would violate the conditions specified. This process helps maintain a record of constraints that may need adjustments.

---

### 3. **Resolving Conflicts**

Once conflicts are detected, the next step is to resolve them. This can involve:

- **Prioritizing Constraints**: Some systems use a priority scheme, where more critical constraints (such as blueprint compliance) are prioritized over others (like soft content preferences). Lower-priority constraints may be relaxed if they conflict with high-priority ones.

- **Applying Conditional Logic**: In cases where conflicts arise from attribute-based constraints, applying logical conditions can help by introducing dependencies. For example, an `INCLUDE` constraint could be relaxed for items in an `ENEMIES` group when an alternative condition (like a high difficulty level) is also met.

- **Error Logging and Feedback**: The system can log conflicts, notifying the user of the conflicting constraints so they can adjust the requirements manually if necessary. This approach is helpful in complex setups where manual intervention is beneficial.

---

### 4. **Types of Conflict Checks and Examples**

#### a. **Friends-Enemies Conflict**

This check ensures that `ALLORNONE` constraints (friends) do not conflict with `ENEMIES` constraints. The conflict check might follow these rules:
- If any item is selected from a friends group, then all items in that group must be selected.
- If an item from a friends group appears in an enemies group, there must be logic to handle cases where both constraints cannot be satisfied simultaneously.

For example, consider the following constraints:

- `C1`: `ALLORNONE` – Items A, B, and C must either all be included or none at all.
- `C2`: `ENEMIES` – Only one of items B or C can be included.

Here, `C1` and `C2` conflict because `C1` requires B and C to be selected together, while `C2` restricts them from appearing together. The conflict detection system would flag this issue.

#### b. **Anchor Conflicts**

Anchor constraints often specify items that are “fixed” across forms. Conflict checks between `ALLORNONE` and `ENEMIES` constraints ensure that anchor items do not violate these rules when they appear in multiple forms.

For example:

- `C3`: `ALLORNONE` – Anchor items D and E must both be present or absent.
- `C4`: `ENEMIES` – Items D and E should not appear together in any form.

This would also raise a conflict, as these two rules cannot be applied simultaneously without violating one of the conditions.

---

### 5. **Conflict Resolution Example**

In practice, here’s how the system might resolve a typical conflict between `INCLUDE` and `EXCLUDE` constraints:

| CONSTRAINT_ID | TYPE     | WHAT  | CONDITION           | LB | UB | ONOFF |
|---------------|----------|-------|---------------------|----|----|-------|
| C5            | INCLUDE  | Item  | `ID == 'Q1'`        |    |    | ON    |
| C6            | EXCLUDE  | Item  | `ID == 'Q1'`        |    |    | ON    |

Since item `Q1` cannot simultaneously be included and excluded, the system would:

- Flag this conflict for manual review, notifying the user.
- Log the conflict for debugging or documentation purposes.
- Potentially prioritize one constraint over the other, based on settings, or disable one if required.

---

### Best Practices for Avoiding Conflicts

- **Unique Item Grouping**: Avoid assigning items to conflicting groups (e.g., do not place the same item in both `ENEMIES` and `ALLORNONE` constraints).
- **Careful Condition Design**: When creating attribute-based conditions, ensure they are distinct and non-overlapping whenever possible.
- **Logical Grouping**: Group items logically based on content, difficulty, and test specifications to reduce the chance of contradictory rules.


### Summary

Conflict handling in constraints is crucial for successful automated test assembly. By systematically detecting, logging, and resolving conflicts, the system can maintain a consistent and feasible set of requirements for test forms. Properly defining constraints, reviewing conflicts, and using conditional logic where appropriate help minimize potential conflicts, allowing for smoother and more reliable form assembly.

---

### Notes on Usage
1. **Testing Configurations**: Before full-scale assembly, test the constraints configuration on a small subset of items to ensure it behaves as expected.
2. **Prioritizing Constraints**: Since this system doesn’t use weights, prioritize constraints by their order in the CSV if necessary.
3. **Debugging Tips**:
   - Use unique `CONSTRAINT_ID`s for easy tracking.
   - Use the `ONOFF` column to toggle constraints without needing to delete rows.
   - When debugging complex conditions, simplify conditions in the `CONDITION` column to verify individual parts.
