from smolagents import Tool


class SuperheroPartyThemeTool(Tool):
    name = "superhero_party_theme_generator"
    description = (
        "Suggests creative superhero-themed party ideas based on a category."
    )
    inputs = {
        "category": {
            "type": "string",
            "description": (
                "The type of superhero party, such as 'classic heroes', "
                "'villain masquerade', or 'futuristic Gotham'."
            ),
        }
    }
    output_type = "string"

    def forward(self, category: str) -> str:
        themes = {
            "classic heroes": (
                "Justice League Gala: Guests come dressed as their favorite DC "
                "heroes with themed cocktails like 'The Kryptonite Punch'."
            ),
            "villain masquerade": (
                "Gotham Rogues' Ball: A mysterious masquerade where guests dress "
                "as classic Batman villains."
            ),
            "futuristic gotham": (
                "Neo-Gotham Night: A cyberpunk-style party inspired by Batman "
                "Beyond, with neon decorations and futuristic gadgets."
            ),
        }
        return themes.get(
            category.lower(),
            "Themed party idea not found. Try 'classic heroes', "
            "'villain masquerade', or 'futuristic Gotham'.",
        )
