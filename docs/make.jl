using Documenter
using DocumenterCitations
using AnimalModels

DocMeta.setdocmeta!(AnimalModels, :DocTestSetup, :(using AnimalModels); recursive=true)

bib = CitationBibliography(joinpath(@__DIR__, "references.bib"); style=:authoryear)

makedocs(;
    modules=[AnimalModels],
    checkdocs=:exports,
    sitename="AnimalModels.jl",
    repo="https://github.com/EvoArt/AnimalModels.jl/blob/{commit}{path}#{line}",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://EvoArt.github.io/AnimalModels.jl",
        repolink="https://github.com/EvoArt/AnimalModels.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Fitting models" => "usage.md",
        "Example: wheat" => "wheat.md",
        "Algorithms" => "algorithms.md",
        "Performance" => "performance.md",
        "Validation" => "validation.md",
        "API reference" => "api.md",
        "References" => "references.md",
    ],
    plugins=[bib],
)

deploydocs(; repo="github.com/EvoArt/AnimalModels.jl")
