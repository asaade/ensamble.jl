Here is the revised version of the `README.md` with the requested additions:

---

# Ensamble.jl

**Automatic Test Assembly (ATA) in Julia using Mixed Integer Programming (MIP)**

## Overview

Ensamble.jl automates the creation of multiple test forms, adhering to constraints such as content balance and difficulty levels, ensuring fairness and consistency across tests. The system is built in Julia, leveraging **Item Response Theory (IRT)** and **Mixed Integer Programming (MIP)** to optimize item selection and assembly.

**Key Features**:
- **Efficient test assembly**: Automates the process of creating multiple test forms.
- **Customizable constraints**: Supports flexible test specifications and item constraints.
- **Multiple solver support**: Compatible with both open-source and commercial solvers.

---

## Why Automatic Test Assembly?

Manual test assembly is time-consuming and prone to errors, especially when creating multiple test forms. Ensamble.jl automates this by:
- **Automating item selection**: Based on predefined constraints.
- **Fairness and balance**: Ensures comparable difficulty levels and content distribution.
- **Providing flexibility**: In constraint handling, supporting a wide range of test designs and statistical properties.

---

## Why Julia and JuMP?

**Julia** is a high-performance programming language that excels in scientific computing and optimization tasks. It combines the ease of writing high-level code with execution speeds close to lower-level languages like C or Fortran. This makes Julia ideal for test assembly processes, which involve heavy computation.

**JuMP** is a domain-specific language for mathematical optimization embedded in Julia. It allows users to formulate complex optimization models in a flexible, high-level manner while interfacing with various solvers. For Ensamble.jl, JuMP manages the optimization of test item selection under predefined constraints.

---

## Methodology

Ensamble.jl uses dichotomous **Item Response Theory (IRT)** models with **1, 2, or 3 parameters**.

These models are used to predict item performance and ensure that test forms meet the required statistical properties.

The methodology implemented in Ensamble.jl is based on the techniques described in **Wim van der Linden's** seminal work, *Linear Models for Optimal Test Design* (2005). This book provides a comprehensive framework for optimal test assembly using linear programming models.

---

## Alternatives

Several other tools exist for automatic test assembly, particularly in R, such as:
- [TestDesign](https://cran.r-project.org/web/packages/TestDesign/index.html)
- [eatATA](https://cran.r-project.org/web/packages/eatATA/index.html)
- [catR](https://cran.r-project.org/web/packages/catR/index.html)
- [ATA.jl (Julia)](https://giadasp.github.io/ATA.jl/docs/)

These tools offer similar features, but Ensamble.jl stands out by providing faster performance through Julia and more flexibility using JuMP for optimization.

---

## Structure

```text
.
├── data
│  ├── anchor.csv
│  ├── config.toml
│  ├── constraints.csv
│  ├── items.csv
│  ├── model.lp
│  └── solver_config.toml
├── docs
│  ├── structure.md
├── LICENSE
├── Manifest.toml
├── Project.toml
├── README.md
├── results
│  ├── combined_plot.pdf
│  ├── forms.csv
│  ├── model.lp
│  ├── results.csv
│  └── tcc_output.csv
├── src
│  ├── ensamble.jl
│  ├── configuration.jl
│  ├── constants.jl
│  ├── config
│  │  ├── assembly_config_loader.jl
│  │  ├── bank_data_loader.jl
│  │  ├── config_loader.jl
│  │  ├── irt_data_loader.jl
│  │  └── validation.jl
│  ├── display
│  │  ├── charts.jl
│  │  └── display_results.jl
│  ├── model
│  │  ├── constraints.jl
│  │  ├── criteria_parser.jl
│  │  ├── model_initializer.jl
│  │  └── solvers.jl
│  └── utils
│     ├── custom_logger.jl
│     ├── stats_functions.jl
│     └── string_utils.jl
└── tests
   └── test_parser.jl
```

---

## How It Works

Ensamble.jl integrates three main components:

1. **Item Response Theory (IRT)**: Models item parameters to predict how well items perform across ability levels.
2. **Mixed Integer Programming (MIP)**: Optimizes item selection to meet test constraints like content balance, test length, item overlap, anchor usage, and statistical properties.
3. **Solvers**: Ensamble.jl supports several solvers to perform MIP optimization, offering flexibility based on available resources.

---

## Supported Solvers

Ensamble.jl supports multiple solvers, giving users flexibility in choosing the best tool for their needs:

1. **[IBM CPLEX](https://www.ibm.com/analytics/cplex-optimizer)**: A powerful commercial solver for large-scale, complex problems.
2. **[CBC (Coin-OR)](https://coin-or.github.io/Cbc/)**: A widely-used open-source solver for smaller applications.
3. **[SCIP](https://scipopt.org/)**: An open-source solver suited for constraint programming.
4. **[GLPK](https://www.gnu.org/software/glpk/)**: Free and open-source, though less efficient for large problems.
5. **[HiGHS](https://highs.dev/)**: An open-source solver optimized for high-performance MIP and LP problems.

---

## Features

- **Multiple Form Assembly**: Generate multiple forms simultaneously, ensuring comparability.
- **Flexible Constraints**: Define custom content constraints, test lengths, and item properties.
- **Solver Integration**: Supports multiple solvers, providing flexibility in optimization.
- **Reports and Visualizations**: Automatically generates detailed reports, including characteristic curves and information curves.
- **Optimization Algorithms**: Built on robust optimization frameworks, Ensamble.jl efficiently handles large item banks and complex constraints.

---

## Usage

### 1. Prepare the Item Bank

Ensure your item bank contains IRT parameters (e.g., difficulty, discrimination, guessing). Calibration tools like **Winsteps**, **ConQuest**, **Bilog**, **Parscale**, or R packages like [TAM](https://cran.r-project.org/web/packages/TAM/index.html) and [mirt](https://cran.r-project.org/web/packages/mirt/index.html) can be used for this purpose.

### 2. Define Test Constraints

Configure your test constraints in the `config.toml` and `contraints.csv` files, including:
- Number of forms
- Test length (min/max)
- Content areas (e.g., math, reading)
- Use of anchor items
- Solver to uses
- Method (Test Characteristic Curve equating, Test Information Curve, Information maximization, etcetera)

### 3. Run Ensamble.jl

To run the test assembly process:

```julia
# Load the configuration and item bank
include("src/ensamble.jl")
using .Ensamble

# Run the test assembly
results = Ensamble.assemble_tests("data/config.toml")
```

---

## Contributing

We welcome contributions! Please submit pull requests or open issues on our [GitHub page](https://github.com/).

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

## Acknowledgments

Inspired by **Wim van der Linden's** *Linear Models for Optimal Test Design* (Springer, 2005), this project was developed to address the need for automated, optimal test assembly in educational and certification assessments.

Thanks to all contributors and users of Ensamble.jl for supporting this project.

---

### Additional Links

- [Julia Programming Language](https://julialang.org/)
- [JuMP: Modeling Language for Mathematical Optimization](https://jump.dev/)

---

This final version eliminates redundancies, includes information on Julia, JuMP, and IRT models, and acknowledges the foundational methodology from Wim van der Linden's work.
