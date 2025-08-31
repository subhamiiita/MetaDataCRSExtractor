# metadata_extractor.py

import json
import openai
from typing import Dict, Any, Tuple

class MetadataExtractor:
    def __init__(self, gpt_api_key: str):
        self.client = openai.OpenAI(api_key=gpt_api_key)
        self.counter = 0
    
    def extract_metadata_and_sentiment(self, user_text: str) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Extract metadata and sentiment from user input"""
        self.counter += 1
        expected_key = f"{self.counter}_user_dialogue"

        prompt = f"""
You are an expert at extracting movie metadata and analyzing sentiment.
Output **only** valid JSON with two keys:

"metadata": one flat dictionary —
• If greeting (hi, hello, good morning, etc.): {{"greet": "<exact words>"}}
• If movie metadata (genre, director, actor, year): use those keys
• Extract ALL mentioned entities, including those with negative context
• Multiple values: comma-separated string in **exact order mentioned**
• If no metadata and not greeting: {{"review": "<direct summary>"}}
• If descriptive content exists alongside metadata: include both genre AND review fields

"sentiment": one dictionary
• Key: "{expected_key}"
• Value:
  - For greetings: ALWAYS "neutral"
  - For metadata: comma-separated sentiment list matching **exact order** of metadata values

CRITICAL EXTRACTION RULES:
• Extract ALL entities mentioned, even with "not", "but not", "except"
• "I like horror but not romantic" → extract BOTH: "horror, romantic"
• "I want action not comedy" → extract BOTH: "action, comedy"
• Sentiment reflects the user's feeling toward each entity in order

SENTIMENT ANALYSIS RULES:
• "I like horror but not romantic" → sentiments: "like, dislike"
• "I want action not comedy" → sentiments: "like, dislike"
• "I love Nolan but hate Tom Cruise" → sentiments: "like, dislike"

Examples:
• "I like horror but not romantic" → {{"genre": "horror, romantic"}}, sentiment: "like, dislike"
• "I want Christopher Nolan films but not with Tom Cruise" → {{"director": "Christopher Nolan", "actor": "Tom Cruise"}}, sentiment: "like, dislike"
• "I love action and comedy movies" → {{"genre": "action, comedy"}}, sentiment: "like, like"

SUMMARY RULES for "review" field:
• Write a direct, concise summary of movie preferences (8-12 words)
• Don't start with "User seeks" or "User wants" - be direct
• Focus on the movie type/characteristics they're looking for

GENRE EXTRACTION RULES:
• "action thriller" → extract as "action, thriller"
• "romantic comedy" → extract as "romance, comedy"
• "sci-fi horror" → extract as "sci-fi, horror"
• "action adventure" → extract as "action, adventure"
• "drama thriller" → extract as "drama, thriller"
and so on for other combinations

VALID GENRES ONLY:
• Extract only these as genres: action, thriller, horror, comedy, drama, romance, sci-fi, fantasy, adventure, crime, mystery, documentary, animation, musical, western, war, biography, family, psychological, superhero
• NOT genres: "Marvel", "DC", "Disney" (these are production companies)
• NOT genres: "character development", "performances", "storylines", "themes", "cinematography", "dialogue", "special effects"
• Production companies and movie characteristics go in review field

EXPLICIT EXTRACTION ONLY:
• Only extract genres that are explicitly mentioned using the exact words from the valid genre list
• Do NOT infer or assume genres based on descriptive content
• "character growth" does NOT automatically mean "drama" genre
• "love interest" does NOT automatically mean "romance" genre
• If descriptive content suggests a genre but doesn't use the exact genre word, put it in review only

INVALID INFERENCE EXAMPLES:
• "emotional scenes" ≠ "drama" (unless "drama" is explicitly said)
• "love story" ≠ "romance" (unless "romance" is explicitly said)
• "scary movie" ≠ "horror" (unless "horror" is explicitly said)
• "funny film" ≠ "comedy" (unless "comedy" is explicitly said)
• Only extract if user literally says the genre word

EXTRACTION PRIORITY:
• Always extract valid genres when mentioned
• Always include review field when descriptive content exists (character development, performances, themes, etc.)
• Both genre and review can coexist in same response
• Example: "psychological movies with character development" → {{"genre": "psychological", "review": "Character-driven psychological films"}}, sentiment: "like, like"
• Example: "Marvel superhero films with good dialogue" → {{"genre": "superhero", "review": "Marvel films with quality dialogue"}}, sentiment: "like, like"

SENTIMENT ALIGNMENT RULES (generic):
• Build the sentiment list so its length equals the TOTAL number of metadata entities

How to count entities:
1. For every metadata key:
   - Keys with comma-separated values (genre, actor, director, year)
     → each comma-separated item = 1 entity
     → e.g. "action, comedy, drama" = 3 entities
   - Keys with a single value (greet, review)
     → always count as 1 entity

2. Sum the entities from all keys to get N

3. Provide exactly N comma-separated sentiment words
   - Order must mirror the order the entities appear (left-to-right across keys, then within each value)

Examples:
• {{"genre": "horror, romance"}} → 2 entities → sentiment: "like, dislike"
• {{"genre": "superhero, action", "review": "Marvel films with good dialogue"}}
  → 2 + 1 = 3 entities → sentiment: "like, like, like"
• {{"actor": "Tom Cruise"}} → 1 entity → sentiment: "like"

Sentiment word choice:
• Use only: like, dislike, neutral (lowercase)
• Determine each word independently by context (negations, positive verbs, etc.)

DEDUPLICATION RULES:
• Remove duplicate entries from comma-separated values
• "superhero, superhero, action, action" → "superhero, action"
• Extract each unique entity only once, regardless of how many times mentioned

REVIEW CONTENT RULES:
• Review should describe what user WANTS, not what they dislike
• "I love X but hate Y" → review should focus on "X without Y"
• "I enjoy Marvel films but not cheesy dialogue" → "Marvel films without cheesy dialogue"
• Always frame review positively toward user preferences

REVIEW FIELD EXCLUSION:
• Do NOT create review field if input only contains valid genres/directors/actors with no additional descriptive content
• "I like action movies" → only extract genre: "action" (no review needed)
• "I like action movies with great stunts" → extract genre: "action" + review: "Movies with great stunts"
• Review field should only exist when there are movie characteristics beyond basic metadata

EXTRACTION SIMPLIFICATION:
• For complex inputs, extract only UNIQUE genres mentioned
• "Marvel and DC superhero films" → genre: "superhero" (only once)
• "action sequences" → genre: "action" (only once)
• Final genre format: "superhero, action" (2 unique genres)
• Review captures production companies and characteristics
• Total entities = unique genres + review field

SENTIMENT LOGIC FIX:
• User enjoys superhero films → "like"
• User enjoys action → "like" 
• User wants films without bad elements → "like" (for review)
• Don't create negative sentiments for genres the user actually likes
• Negative aspects go into review phrasing, not genre sentiment

Remember:
• Extract ALL entities mentioned, regardless of positive/negative context
• Always include review field when descriptive movie characteristics are mentioned
• Genre and review fields can both exist in same response
• Production companies (Marvel, DC, Disney) are NOT genres - put in review
• Movie characteristics (dialogue, effects, cinematography) are NOT genres - put in review
• Use only lowercase like/dislike/neutral
• Sentiment count must EXACTLY match total metadata entity count
• Remove duplicates from genre lists
• Don't assign negative sentiment to genres user actually likes
• ONLY extract genres when exact genre words are explicitly mentioned
• Do NOT infer genres from descriptive content
• NEVER wrap JSON in triple back-ticks

User: "{user_text}"
"""






        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert at extracting movie metadata and analyzing sentiment. For greetings, always return 'neutral' sentiment. For movie metadata, analyze each entity's sentiment carefully. Return only JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=250,
        )

        raw_json = response.choices[0].message.content.strip()
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Bad GPT JSON: {raw_json}") from e

        # Minimal sanity checks
        if "metadata" not in data or "sentiment" not in data:
            raise RuntimeError("GPT response missing required keys.")

        return data["metadata"], data["sentiment"]
    
    def process_input(self, user_input: str) -> None:
        """Process user input and display results"""
        if not user_input.strip():
            print("❌ Empty input")
            return
        
        try:
            metadata, sentiment = self.extract_metadata_and_sentiment(user_input)
            print(f"🗂️ Metadata extracted: {metadata}")
            print(f"❤️ Sentiment tag: {sentiment}")
        except Exception as e:
            print(f"❌ Error processing input: {e}")

def main():
    # Initialize with your GPT API key
    api_key = "OPEN_AI_API_KEY"
    if not api_key:
        print("❌ API key required")
        return
    
    extractor = MetadataExtractor(api_key)
    
    print("🎬 Movie Metadata & Sentiment Extractor")
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        extractor.process_input(user_input)
        print()  # Empty line for readability

if __name__ == "__main__":
    main()
