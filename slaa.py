import requests
 
API_URL = "https://api-inference.huggingface.co/models/suno/bark"
headers = {"Authorization": "Bearer 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"}
 
def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.content
 
audio_bytes = query({
	"inputs": "The answer to the universe is 42",
})
# You can access the audio with IPython.display for example
from IPython.display import Audio
Audio(audio_bytes)