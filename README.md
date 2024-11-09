### Ensamble.jl

**Efficient Automated Test Assembly (ATA) in Julia using Mixed Integer Programming (MIP)**

* * *

Ensamble.jl is a Julia-based ATA solution that integrates **Item Response Theory (IRT)** with **Mixed Integer Programming** to generate comparable, balanced test forms. The system accommodates multiple test assembly constraints, ensuring content balance, difficulty, and fairness across test forms.

### Key Features

  - **Optimized Test Assembly**: Facilitates the automated creation of standardized test forms.
  - **Flexible Constraint Management**: Enables custom constraints for item selection, including content, difficulty, and anchor items.
  - **Multi-Solver Compatibility**: Supports CPLEX, Gurobi, HiGHS, SCIP, and GLPK for flexible solver choice.

### Main Components

 1. **IRT Framework**: Models item performance by aligning with ability levels.
 2. **MIP Solver Integration**: Optimizes item selection with defined constraints.
 3. **Adaptable Configuration**: `config.toml` and `constraints.csv` specify item parameters and assembly rules, including:

      + Number of forms and test length
      + Content balance and difficulty requirements
      + Solver choice and assembly methods (e.g., TCC, TIC, Mixed).


### Why Julia and JuMP?

**Julia** is a high-performance programming language that excels in scientific computing and optimization tasks. It combines the ease of writing high-level code with execution speeds close to lower-level languages like C or Fortran. This makes Julia ideal for test assembly processes, which involve heavy computation and avoid the need to mix different programming languages, as is done in R and Python..

**JuMP** is a rich domain-specific language for mathematical optimization embedded in Julia. It allows users to formulate complex optimization models in a flexible, high-level manner while interfacing with various solvers. For Ensamble.jl, JuMP manages the optimization of test item selection under predefined constraints.

### Supported Assembly Methods

#### Test Characteristic Curve (TCC)

TCC. Matches the test characteristic curve of selected items to a reference test, ensuring comparable scores across forms, a type of equating observed scores 'pre-equating'. When all items are dichotomous, the method is extended following van de Linden's suggestion to also equate the curve of powers of the probability of correct answer of the items (similar to his "local equating" of observed scores).

TCC2. Matches the test characteristic curve and the variance of the selected items in the forms.

MIXED. Matches the test characteristic curve and the information curve of the the forms.

#### Test Information Curve (TIC)

TIC. Optimizes measurement precision by matching test information to predefined targets at specific ability levels.

TIC2. Maximizes the information of all forms at selected points of the ability scale.

TIC3. Selects items to maximize information at potentially different points of the ability scale.

* * *

### Configuration File Documentation

The TOML configuration file defines parameters for test assembly settings, item constraints, file paths, and solver options. It consists of several key sections:

#### `[FORMS]`

  - **NumForms**: Sets the number of forms to be assembled.
  - **AnchorTests**: Specifies the number of anchor forms, used cycling across all forms to ensure consistency.
  - **ShadowTest**: Indicates the use of a 'shadow test' for iterative assembly. Shadow tests are used as an heuristic method to maintain constraints across multiple assembly cycles by reserving items for future forms.

#### `[IRT]`

  - **METHOD**: Specifies the method for scoring or matching (e.g., `TCC2`). Determines the model's scoring approach.
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

### Configuration Examples

#### Example `config.toml`

```toml
[FORMS]
NumForms = 4       # New forms to assemble
AnchorTests = 2    # Number of anchor test forms in the anchor test file
ShadowTest = 1     # Use a shadow test

[IRT]
METHOD = "TCC"                        # Test Characteristic Curve matching
THETA = [-2.5, -0.5, 0.0, 0.5, 2.5]   # Points in the theta scale to confirm the match
TAU = []                              # Value of the expected scores from the reference test at THETA. If empty,
                                      # the values are test estimated scores as an average from the items in the bank
TAU_INFO = []                         # Value of the test information at THETA. They are estimated, but not needed here.
R = 3                                 # Number of powers of the probability to compare. Only used with dichotomous items.
D = 1.0                               # Constant

[FILES]
# Inputs
ITEMSFILE = "data/items.csv"
ANCHORFILE = "data/anchor.csv"
CONSTRAINTSFILE = "data/constraints.csv"
# Outputs
RESULTSFILE = "results/results.csv"
FORMSFILE = "results/forms.csv"
TCCFILE = "results/tcc_output.csv"
PLOTFILE = "results/plot_output.png"

# Special reports
REPORTCATEGORIES = ["CONTENT", "LEVEL"] # Reports counts per form on these columns of the bank
REPORTSUMS = ["WORDS", "IMAGES"]        # Reports sums per form on these columns

# Other
SOLVER = "cplex"  # An open source performat option is "highs". Any solver shoud be installed separatedly
VERBOSE = 1       # Three levels of detail in printed information
```

#### Constraint configuration `constraints.csv`

Ensamble.jl includes facilities to modify the constraints used to assemble the tests with a format inspired by the [TestDesign R Package](https://cran.r-project.org/package=TestDesign), An R package designed for automated test assembly and item selection for educational testing and psychometrics.

It uses a CSV. Here is an example:

| CONSTRAINT_ID | TYPE      | CONDITION        | LB | UB | ONOFF |
|:------------- |:--------- |:---------------- |:-- |:-- |:----- |
| C1            | TEST      |                  | 30 | 30 | ON    |
| C2            | NUMBER    | LEVEL == 3       | 10 | 10 | ON    |
| C3            | SUM       | WORDS, LEVEL==1  | 50 | 80 | ON    |
| C4            | ALLORNONE | ID IN [101, 102] |    |    | ON    |
| C5            | ENEMIES   | ENEMIES          |    |    | ON    |

  - `C1` (Test Length): Specifies a total of 30 items per form.
  - `C2` (Content Balance): Requires exactly 10 items at Level 3.
  - `C3` (Conditional Sum): Ensures that items meeting `LEVEL == 1` contribute a total of 50–80 words.
  - `C4` (All-or-None): Items 101 and 102 must be either both included or excluded in any form.
  - `C5` (Enemies): Items in column ENEMIES are not to be paired with items with the same value/group in that column.

* * *

### Running the Assembly Process

 1. **Prepare Item Bank**: Provide item bank data in CSV format, including IRT parameters.

 2. **Define Constraints**: Set test assembly rules in `config.toml` and `constraints.csv`.
 3. **Run Assembly**:

    ```julia
    include("src/ensamble.jl")
    using .Ensamble
    results = Ensamble.assemble_tests("data/config.toml")
    ```

### Available Files and Output

  - **Results**: Check `results.csv`, `forms.csv`, `tcc_output.csv`, and `plot_output.png` for outputs.
  - **Documentation**: Detailed descriptions available in `docs/structure.md`.

* * *

### Supported Solvers

Ensamble.jl works with multiple solvers for flexibility:

  - **CPLEX** (IBM)
  - **HiGHS**
  - **SCIP**
  - **GLPK**

To use a specific solver, adjust the `SOLVER` entry in `config.toml` accordingly.
