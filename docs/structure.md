.
├── data
│  ├── anchor.csv
│  ├── config.toml
│  ├── constraints.csv
│  ├── items.csv
│  ├── model.lp
│  └── solver_config.toml
├── debug.log
├── docs
│  └── structure.md
├── items.csv
├── LICENSE
├── logfile.log
├── Manifest.toml
├── Project.toml
├── README.md
├── results
│  ├── combined_plot.pdf
│  ├── model.lp
│  ├── p1_characteristic_curves.svg
│  ├── p2_information_curves.svg
│  ├── p3_1_observed_scores_n01.svg
│  ├── p3_2_observed_scores_n_variations.svg
│  ├── results.csv
│  └── tcc.csv
├── src
│  ├── config
│  │  ├── assembly_config_loader.jl
│  │  ├── bank_data_loader.jl
│  │  ├── config_loader.jl
│  │  └── irt_data_loader.jl
│  ├── configuration.jl
│  ├── constants.jl
│  ├── debug.log
│  ├── display
│  │  ├── charts.jl
│  │  └── display_results.jl
│  ├── ensamble.jl
│  ├── logfile.log
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
