using Documenter
using QuantitativeGenetics

DocMeta.setdocmeta!(QuantitativeGenetics, :DocTestSetup,
                   :(using QuantitativeGenetics); recursive=true)

makedocs(;
    modules=[QuantitativeGenetics],
    checkdocs=:exports,
    sitename="QuantitativeGenetics.jl",
    repo="https://github.com/EvoArt/QuantitativeGenetics.jl/blob/{commit}{path}#{line}",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://EvoArt.github.io/QuantitativeGenetics.jl",
        repolink="https://github.com/EvoArt/QuantitativeGenetics.jl",
    ),
    pages=["Home" => "index.md", "Example: wheat" => "wheat.md"],
)

deploydocs(; repo="github.com/EvoArt/QuantitativeGenetics.jl")
