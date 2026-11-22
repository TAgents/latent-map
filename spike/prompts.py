"""Fixed public prompts for the spike. Four topics, five prompts each."""

PROMPTS = {
    "chat": [
        "Hey, how was your weekend?",
        "Thanks so much for helping me yesterday.",
        "Can you recommend a good book for a long flight?",
        "I'm not sure what to cook for dinner tonight.",
        "Sorry I'm late, the train was delayed again.",
    ],
    "code": [
        "Write a Python function that reverses a string.",
        "How do I list files in a directory with bash?",
        "Explain what a Python decorator does.",
        "Write a SQL query to count rows in a table called users.",
        "What does git rebase do?",
    ],
    "dates": [
        "How many days are there from March 3 to July 19?",
        "What month comes after September?",
        "If today is Monday, what day is it in ten days?",
        "How many weeks are in a year?",
        "Which months have exactly thirty days?",
    ],
    "math": [
        "What is 17 times 23?",
        "Convert 5 miles to kilometres.",
        "What is 15 percent of 240?",
        "What is the square root of 144?",
        "Add 3/4 and 2/3.",
    ],
}
