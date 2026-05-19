import requests

# Free NewsAPI key — signup at newsapi.org (takes 2 min)
NEWS_API_KEY = "24ccd3bb5dff4b0098b0642d35d8c2ef"

def get_sentiment():
    try:
        url = (
            f"https://newsapi.org/v2/everything"
            f"?q=copper+zinc+price+metal"
            f"&language=en&pageSize=5"
            f"&apiKey={NEWS_API_KEY}"
        )
        response = requests.get(url).json()
        headlines = [a["title"] for a in response.get("articles", [])]

        if not headlines:
            return "NEUTRAL", [], 0

        # Simple keyword-based sentiment (no heavy AI model needed for demo)
        positive_words = ["rise", "surge", "gain", "high", "strong",
                         "rally", "jump", "demand", "growth", "bullish"]
        negative_words = ["fall", "drop", "low", "weak", "decline",
                         "crash", "cut", "bearish", "slump", "reduce"]

        score = 0
        for h in headlines:
            h_lower = h.lower()
            score += sum(1 for w in positive_words if w in h_lower)
            score -= sum(1 for w in negative_words if w in h_lower)

        if score > 1:
            signal = "BULLISH 📈"
        elif score < -1:
            signal = "BEARISH 📉"
        else:
            signal = "NEUTRAL ➡️"

        return signal, headlines, score

    except Exception as e:
        print(f"News fetch error: {e}")
        return "NEUTRAL", [], 0

if __name__ == "__main__":
    signal, headlines, score = get_sentiment()
    print(f"\nSentiment: {signal} (score: {score})")
    print("\nHeadlines:")
    for h in headlines:
        print(f"  - {h}")