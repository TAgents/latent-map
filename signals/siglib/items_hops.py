"""Two-step questions for the unwritten-step test. Each item has a subject, the
unwritten step ("bridge") and the answer. Facts are well known; the model's own
single-step answers decide which items are kept."""

# family: (two-step question, first-step question, second-step question, rows of (subject, bridge, answer))
FAMILIES = {
    "person_capital": (
        "What is the capital of the country where {s} was born? Answer with the city name only.",
        "In which country was {s} born? Answer with the country name only.",
        "What is the capital of {b}? Answer with the city name only.",
        [("Marie Curie", "Poland", "Warsaw"), ("Wolfgang Amadeus Mozart", "Austria", "Vienna"), ("Pablo Picasso", "Spain", "Madrid"),
         ("Mahatma Gandhi", "India", "New Delhi"), ("Leo Tolstoy", "Russia", "Moscow"), ("Vincent van Gogh", "Netherlands", "Amsterdam"),
         ("Leonardo da Vinci", "Italy", "Rome"), ("Confucius", "China", "Beijing"), ("Frida Kahlo", "Mexico", "Mexico City"),
         ("Pelé", "Brazil", "Brasília"), ("Henrik Ibsen", "Norway", "Oslo"), ("Hans Christian Andersen", "Denmark", "Copenhagen"),
         ("Jean Sibelius", "Finland", "Helsinki"), ("Cleopatra", "Egypt", "Cairo"), ("Socrates", "Greece", "Athens"),
         ("Akira Kurosawa", "Japan", "Tokyo"), ("Jorge Luis Borges", "Argentina", "Buenos Aires"), ("Alfred Nobel", "Sweden", "Stockholm"),
         ("Albert Einstein", "Germany", "Berlin"), ("Samuel Beckett", "Ireland", "Dublin"), ("Pablo Neruda", "Chile", "Santiago")],
    ),
    "landmark_currency": (
        "What currency is used in the country where {s} is located? Answer with the currency name only.",
        "In which country is {s} located? Answer with the country name only.",
        "What currency is used in {b}? Answer with the currency name only.",
        [("the Colosseum", "Italy", "euro"), ("the Taj Mahal", "India", "rupee"), ("the Great Wall", "China", "yuan"),
         ("Big Ben", "United Kingdom", "pound"), ("the Statue of Liberty", "United States", "dollar"), ("Machu Picchu", "Peru", "sol"),
         ("Mount Fuji", "Japan", "yen"), ("the Kremlin", "Russia", "ruble"), ("Petra", "Jordan", "dinar"),
         ("Angkor Wat", "Cambodia", "riel"), ("Table Mountain", "South Africa", "rand"), ("Chichen Itza", "Mexico", "peso"),
         ("the Hagia Sophia", "Turkey", "lira"), ("the Burj Khalifa", "United Arab Emirates", "dirham"),
         ("the Matterhorn", "Switzerland", "franc"), ("Christ the Redeemer", "Brazil", "real"),
         ("the Sydney Opera House", "Australia", "dollar"), ("the Eiffel Tower", "France", "euro"),
         ("the Pyramids of Giza", "Egypt", "pound"), ("the CN Tower", "Canada", "dollar")],
    ),
    "book_country": (
        "In which country was the author of \"{s}\" born? Answer with the country name only.",
        "Who wrote \"{s}\"? Answer with the author's name only.",
        "In which country was {b} born? Answer with the country name only.",
        [("Don Quixote", "Cervantes", "Spain"), ("War and Peace", "Tolstoy", "Russia"), ("Pride and Prejudice", "Austen", "England"),
         ("Les Misérables", "Hugo", "France"), ("The Divine Comedy", "Dante", "Italy"), ("Faust", "Goethe", "Germany"),
         ("One Hundred Years of Solitude", "Márquez", "Colombia"), ("The Old Man and the Sea", "Hemingway", "United States"),
         ("Crime and Punishment", "Dostoevsky", "Russia"), ("The Odyssey", "Homer", "Greece"), ("Ulysses", "Joyce", "Ireland"),
         ("Madame Bovary", "Flaubert", "France"), ("Things Fall Apart", "Achebe", "Nigeria"), ("Oliver Twist", "Dickens", "England"),
         ("Moby-Dick", "Melville", "United States"), ("The Alchemist", "Coelho", "Brazil"), ("Pippi Longstocking", "Lindgren", "Sweden"),
         ("Heidi", "Spyri", "Switzerland"), ("The Kite Runner", "Hosseini", "Afghanistan"), ("Peer Gynt", "Ibsen", "Norway")],
    ),
    "company_capital": (
        "What is the capital of the country where the company {s} was founded? Answer with the city name only.",
        "In which country was the company {s} founded? Answer with the country name only.",
        "What is the capital of {b}? Answer with the city name only.",
        [("Toyota", "Japan", "Tokyo"), ("IKEA", "Sweden", "Stockholm"), ("Nokia", "Finland", "Helsinki"), ("Samsung", "South Korea", "Seoul"),
         ("LEGO", "Denmark", "Copenhagen"), ("Nestlé", "Switzerland", "Bern"), ("Philips", "Netherlands", "Amsterdam"),
         ("Volkswagen", "Germany", "Berlin"), ("Ferrari", "Italy", "Rome"), ("Renault", "France", "Paris"), ("Zara", "Spain", "Madrid"),
         ("Guinness", "Ireland", "Dublin"), ("Huawei", "China", "Beijing"), ("Petrobras", "Brazil", "Brasília"),
         ("Qantas", "Australia", "Canberra"), ("Aeroflot", "Russia", "Moscow"), ("Škoda", "Czech Republic", "Prague"),
         ("Red Bull", "Austria", "Vienna"), ("Tata Motors", "India", "New Delhi")],
    ),
}
# Looked at during the feasibility check, before the criteria were fixed: reported, but left out of the criteria.
SEEN_IN_STEP0 = {("person_capital", "Marie Curie"), ("landmark_currency", "the Colosseum"), ("book_country", "Don Quixote")}


def build() -> list[dict]:
    items = []
    for family, (two, first, second, rows) in FAMILIES.items():
        for subject, bridge, answer in rows:
            items.append({
                "id": f"{family}/{subject}", "family": family, "subject": subject, "bridge": bridge, "answer": answer,
                "two_step": two.format(s=subject), "first_step": first.format(s=subject), "second_step": second.format(b=bridge),
                "seen_in_step0": (family, subject) in SEEN_IN_STEP0,
            })
    return items
