# Ensamble.jl

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

For more details, see the complete documentation in [`docs/structure.md`](docs/structure.md).
