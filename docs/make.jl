using Documenter
using MiddleOutProteinDesign

makedocs(
    sitename = "MiddleOutProteinDesign",
    modules = [MiddleOutProteinDesign],
    pages = [
        "Home" => "index.md",
    ],
)
