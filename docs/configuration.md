### Configuration File Documentation

The TOML configuration file defines parameters for test assembly settings, item constraints, file paths, and solver options. It consists of several key sections:

#### `[FORMS]`

  - **NumForms**: Sets the number of forms to be assembled.
  - **AnchorTests**: Specifies the number of anchor forms, used cycling across all forms to ensure consistency.
  - **ShadowTest**: Indicates the use of a 'shadow test' for iterative assembly. Shadow tests are used as an heuristic method to maintain constraints across multiple assembly cycles by reserving items for future forms.

#### `[IRT]`

  - # Ensamble.jl

**Automated Test Assembly (ATA) in Julia using Mixed Integer Programming (MIP)**

Ensamble.jl is a Julia-based ATA tool that integrates **Item Response Theory (IRT)** with **Mixed Integer Programming**. It generates balanced, comparable test forms under various constraints—ensuring content coverage, difficulty alignment, and fair distribution of items across multiple forms.

## Key Features

- **Optimized Test Assembly**
  Automatically create standardized forms that meet target criteria (e.g., item difficulty, content, test length).

- **Flexible Constraints**
  Handle a wide variety of constraints (content, difficulty, anchor items, and more) through a user-defined configuration.

- **Multi-Solver Compatibility**
  Works with CPLEX, Gurobi, HiGHS, SCIP, GLPK, and others.

- **Shadow Tests & Anchor Items**
  - **Shadow tests** preserve item pools for subsequent assemblies, allowing iterative form-building.
  - **Anchor items** are predefined items appearing in multiple operational forms to ensure score comparability.

- **Reporting**
  Generates charts (Test Characteristic and Information Curves) and summary reports (e.g., number of items by category, anchor usage, expected scores).

## Why Julia + JuMP?

Julia is a high-performance language well-suited for scientific computing and large-scale optimization. Its syntax is close to mathematical notation, making it intuitive for modeling.
**JuMP**, a Julia-based optimization layer, makes constraint definitions straightforward and integrates seamlessly with high-performance solvers (e.g., Gurobi, CPLEX, HiGHS, GLPK).

## Supported Assembly Methods

- **Test Characteristic Curve (TCC)**
  Matches item-based curves to a reference test, enabling consistent expected scores across forms.
- **Test Information Curve (TIC)**
  Maximizes measurement precision at specified ability levels.

Both methods can be combined or extended (e.g., TCC + TIC, multi-point optimization) for specialized needs.

## Configuration Overview

All assembly rules and file paths reside in a `config.toml` file, which includes:

- **[FORMS]**
  Number of forms, anchor test details, and whether shadow tests are used.
- **[IRT]**
  Assembly method (TCC/TIC), the ability levels (`THETA`) to target, and whether to match expected scores or information curves.
- **[FILES]**
  Paths for item banks, anchor items, constraints, and output files.
- **[SOLVER]**
  Desired solver (e.g., `cplex`, `highs`) and verbosity level.

### Example `config.toml`

```toml
[FORMS]
NumForms = 4
AnchorTests = 2
ShadowTest = 1

[IRT]
METHOD = "TCC"
THETA = [-2.5, -0.5, 0.0, 0.5, 2.5]
TAU = []
TAU_INFO = []
R = 3
D = 1.0

[FILES]
ITEMSFILE = "data/items.csv"
ANCHORFILE = "data/anchor.csv"
CONSTRAINTSFILE = "data/constraints.csv"
RESULTSFILE = "results/results.csv"
FORMSFILE = "results/forms.csv"
TCCFILE = "results/tcc_output.csv"
PLOTFILE = "results/plot_output.png"
REPORTCATEGORIES = ["CONTENT","LEVEL"]
REPORTSUMS = ["WORDS","IMAGES"]

SOLVER = "cplex"
VERBOSE = 1
```

## Constraint Format

A CSV file (e.g., `constraints.csv`) defines additional constraints (e.g., test length, item groupings, all-or-none rules). Here’s an example:

| CONSTRAINT_ID | TYPE      | CONDITION        | LB | UB | ONOFF |
|---------------|-----------|------------------|----|----|------|
| C1            | TEST      |                  | 30 | 30 | ON   |
| C2            | NUMBER    | LEVEL == 3       | 10 | 10 | ON   |
| C3            | SUM       | WORDS, LEVEL==1  | 50 | 80 | ON   |
| C4            | ALLORNONE | ID IN [101,102]  |    |    | ON   |
| C5            | ENEMIES   | ENEMIES          |    |    | ON   |

## Running the Assembly

1. **Prepare Items**
   Provide a CSV with item parameters and attributes.
2. **Set Constraints**
   Edit `config.toml` and `constraints.csv`.
3. **Execute in Julia**
   ```julia
   include("src/ensamble.jl")
   using .Ensamble
   results = Ensamble.assemble_tests("data/config.toml")
   ```

## Output Files

- **Results**: `results.csv` and `forms.csv`
- **Plots & TCC Data**: `tcc_output.csv`, `plot_output.png`
- **Reports**: Summaries by category, anchor usage, etc.

## Supported Solvers

- **CPLEX** (commercial)
- **Gurobi** (commercial)
- **HiGHS**, **Cbc**, **SCIP**, **GLPK** (open source)

Adjust `SOLVER` in your config file to switch between them.

---

For more details, see the complete documentation in:
- [`docs/configuration.md`](docs/configuration.md).
- [`docs/constraints.md`](docs/constraints.md).
- [`docs/structure.md`](docs/structure.md).
**METHOD**: Specifies the method for scoring or matching (e.g., `TCC2`). Determines the model's scoring approach.
  - **THETA**: Ability level values to match expected scores and other metrics. I set of 3..5 well targeted theta points in the curve are often enough to match the compete range.
  - **TAU** and **TAU_INFO**: Arrays for target means and variances at specified theta levels for characteristic and information curves.
    
      + TAU can either be an empty array or a set of vectors (e.g., `[[0.5, 0.4], ...]`).. If TAU is emplty, an approximation will be estimated from the items in the bank
  - **RELATIVETARGETWEIGHTS** and **RELATIVETARGETPOINTS**: Lists of weights and target points for information or score matching, providing flexibility in test design goals.
  - **R**: The maximum power for item probabilities of correct response for use using the TCC method, folowing van del Linden's observed scores "local equating" method.
  - **D**: Scaling constant for IRT logistic models (typically `1.0` for standard logistic models and 1.7 to approximate a normat curve).

#### `[FILES]`

  - **ITEMSFILE**: Path to the CSV file with item parameters and attributes.
  - **ANCHORFILE**: Path to the anchor items CSV, if applicable.
  - **CONSTRAINTSFILE**: Path to the constraints CSV, specifying the conditions items must satisfy.
  - **RESULTSFILE**: Path for outputting final test assembly results.
  - **FORMSFILE**: Path to save individual form compositions.
  - **TCCFILE**: Output file for Test Characteristic Curve (TCC) data.
  - **PLOTFILE**: Path to save visual plots for test information and TCC data.
  - **REPORTCATEGORIES** and **REPORTSUMS**: Lists of item attributes to include in reporting (e.g., categories for content analysis and sums for item attributes).

#### Solver and Debugging

  - **SOLVER**: Name of the solver (e.g., `cplex`) used for optimization.
  - **VERBOSE**: Level of verbosity for logging and debugging, where `0` is silent, and higher values increase detail in the output.

## Example configuration

Here is an example of a configuration for assembling tests using the Test Characteristic Curve (TCC) method. This example uses TCC constraints to control the score distribution across test forms:

### Configuration (TOML) for TCC Optimization

```toml
[FORMS]
NumForms = 5
AnchorTests = 2
ShadowTest = 1

[IRT]
METHOD = "TCC"
THETA = [-3.0, -1.0, 0.0, 1.0, 3.0]
TAU = [[0.5, 0.4, 0.6], [0.6, 0.5, 0.7], [0.7, 0.6, 0.8], [0.8, 0.7, 0.9]]
TAU_INFO = [0.3, 0.4, 0.5, 0.6, 0.7]
RELATIVETARGETWEIGHTS = [1, 1, 1, 1, 1]
RELATIVETARGETPOINTS = [-1.0, 0.0, 1.0]
R = 3
D = 1.0

[FILES]
ITEMSFILE = "data/items.csv"
ANCHORFILE = "data/anchor.csv"
CONSTRAINTSFILE = "data/constraints.csv"
RESULTSFILE = "results/results.csv"
FORMSFILE = "results/forms.csv"
TCCFILE = "results/tcc_output.csv"
PLOTFILE = "results/plot_output.png"
REPORTCATEGORIES = ["CATEGORY_A", "CATEGORY_B"]
REPORTSUMS = ["WORD_COUNT", "IMAGE_COUNT"]

SOLVER = "cplex"
VERBOSE = 1
```

### Example of `items.csv` File Format

The `items.csv` file should contain columns with item parameters, typically including:

  - `ID`: Unique identifier for each item
  - `MODEL_TYPE`: Specifies model type (e.g., "2PL", "3PL", "PCM")
  - `A`: Discrimination parameter
  - `B`: Difficulty (or threshold) parameters, usually as multiple columns `B1`, `B2`, `B3` for polytomous items
  - `C`: Guessing parameter (for "3PL" model)
  - `NUM_CATEGORIES`: Number of response categories for polytomous items

Example `items.csv`:

```csv
ID,MODEL_TYPE,A,B1,B2,B3,C,NUM_CATEGORIES
item1,2PL,1.0,-1.0,,
item2,3PL,0.8,0.5,,0.2
item3,PCM,1.2,0.5,1.0,1.5,,4
item4,GPCM,0.9,0.3,1.2,,
```

For effective assembly, verify that the items and constraints align with the test blueprint requirements.
