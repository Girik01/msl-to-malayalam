"""
Generate pseudo-parallel gloss→Malayalam sentence pairs.
Paper Section III-F: pseudo-parallel data compensates for missing
annotated gloss-Malayalam datasets.

Uses the 61 MSL alphabet class names as the gloss vocabulary.
"""

import os, json, time
from deep_translator import GoogleTranslator

BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT     = os.path.join(BASE, "checkpoints", "pseudo_parallel.json")

# Gloss vocabulary must match Sahaayi class names exactly
PAIRS = [
    (["Hello"],                      "Hello"),
    (["Hello", "Welcome"],           "Hello welcome"),
    (["Bye"],                        "Goodbye"),
    (["Good"],                       "Good"),
    (["Happy"],                      "I am happy"),
    (["Sa", "Ha", "Aa", "Ya", "Ma"], "Help"),
    (["Na", "Ma", "Sa", "Ka", "Ra", "Ma"], "Greetings"),
    (["Ka", "Aa", "Na", "U"],        "Please wait"),
    (["Va", "Aa"],                   "Come"),
    (["Pa", "O", "Ka", "Ka", "U"],   "Go"),
    (["Ka", "Zha", "I", "Ka", "Ka", "U"], "Eat"),
    (["Ka", "U", "Di", "Ka", "Ka", "U"],  "Drink water"),
    (["Na", "Aa", "Na"],             "I"),
    (["Na", "I", "Nga", "Na", "Aa", "La"], "You"),
    (["Va", "I", "Du"],              "Home"),
    (["Pa", "Tha", "Aa", "La", "Ya", "Ma"], "School"),
    (["Sa", "Ha", "Aa", "Ya", "I", "Ka", "Ka", "U"], "Please help"),
    (["Na", "Aa", "Na", "Pa", "O", "Ka", "Ka", "U"], "I will go"),
    (["Na", "I", "Nga", "Na", "Va", "Aa"], "You come"),
    (["Na", "Aa", "Na", "Ka", "Zha", "I", "Ka", "Ka", "U"], "I will eat"),
    (["Tha", "Na", "Na", "Ya", "Va", "A", "Da"], "Thank you"),
    (["Ma", "Na", "Sa", "I", "La", "Aa", "Ya", "I"], "I understand"),
    (["I", "La", "La"],              "No"),
    (["U", "N", "Tha"],              "Yes"),
    (["Va", "E", "La", "I"],        "Work"),
    (["Pa", "Tha", "I", "Va", "U", "Sa", "Ra", "A", "Ma", "A", "Ra", "Aa", "N", "U"],
                                     "Wait a moment"),
    (["Va", "I", "Du", "Ma", "Aa", "Na", "U"],  "I am at home"),
    (["A", "Dhu", "Aa", "N", "U"],   "That is it"),
    (["I", "Dhu", "Aa", "N", "U"],   "This is it"),
    (["Aa", "Ya", "I"],              "Came"),
    (["Pa", "O", "Ya", "I"],         "Gone"),
]


def generate():
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} pairs from {OUT}")
        return data

    print(f"Generating {len(PAIRS)} pseudo-parallel pairs...")
    t = GoogleTranslator(source="en", target="ml")
    data = []
    for i, (gloss, english) in enumerate(PAIRS):
        try:
            ml = t.translate(english)
            data.append({"gloss": gloss, "english": english, "malayalam": ml})
            print(f"  [{i+1}/{len(PAIRS)}] {english} → {ml}")
            time.sleep(0.3)
        except Exception as e:
            print(f"  SKIP: {english} ({e})")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(data)} pairs → {OUT}")
    return data


if __name__ == "__main__":
    generate()
