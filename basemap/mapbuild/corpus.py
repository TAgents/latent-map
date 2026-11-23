"""A small grid of topics and text forms for the model to write about."""

TOPICS = [
    "volcanoes", "the French Revolution", "black holes", "baking bread", "chess openings", "the immune system",
    "interest rates", "jazz", "the Roman Empire", "photosynthesis", "sorting algorithms", "ocean tides",
    "the stock market", "marathon training", "prime numbers", "climate change", "the printing press",
    "sleep and dreams", "electric cars", "database indexes", "the Olympic Games", "vaccines", "coffee", "bridges",
]

FORMS = {
    "explain": "Explain {topic} in one clear paragraph.",
    "dialogue": "Write a short dialogue between two friends talking about {topic}.",
    "code": "Write a short Python function related to {topic}, with comments.",
    "story": "Write the opening of a short story in which {topic} matters.",
    "steps": "Give a numbered list of five practical tips or facts about {topic}.",
}


def build_prompts() -> list[dict]:
    return [
        {"id": f"{t:02d}-{form}", "topic": topic, "form": form, "prompt": template.format(topic=topic)}
        for t, topic in enumerate(TOPICS)
        for form, template in FORMS.items()
    ]
