param(
    [ValidateSet("core-ph", "expanded-ph", "expanded-variants")]
    [string]$Scope = "expanded-ph",
    [string]$LocalDir = "data/robomimic"
)

$ErrorActionPreference = "Stop"

$corePh = @(
    "v1.5/lift/ph/low_dim_v15.hdf5",
    "v1.5/can/ph/low_dim_v15.hdf5",
    "v1.5/square/ph/low_dim_v15.hdf5"
)

$expandedPh = $corePh + @(
    "v1.5/tool_hang/ph/low_dim_v15.hdf5",
    "v1.5/transport/ph/low_dim_v15.hdf5"
)

$expandedVariants = $expandedPh + @(
    "v1.5/lift/mh/low_dim_v15.hdf5",
    "v1.5/lift/mg/low_dim_v15.hdf5",
    "v1.5/can/mh/low_dim_v15.hdf5",
    "v1.5/can/mg/low_dim_v15.hdf5",
    "v1.5/square/mh/low_dim_v15.hdf5",
    "v1.5/transport/mh/low_dim_v15.hdf5"
)

$files = switch ($Scope) {
    "core-ph" { $corePh }
    "expanded-ph" { $expandedPh }
    "expanded-variants" { $expandedVariants }
}

hf download robomimic/robomimic_datasets @files --repo-type dataset --local-dir $LocalDir
