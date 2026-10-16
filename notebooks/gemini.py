from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client()

interaction = client.interactions.create(
    model="gemini-3.8-flash",
    input="who is PM Modi and what is his contribution to India?"
)

print(interaction.output_text)


