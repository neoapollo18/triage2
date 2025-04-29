import requests

url = "https://triage2-production-ac73.up.railway.app/predict"  # Your deployed API endpoint

payload = {
  "business": {
    "business_id": "bus123"
  },
  "threepl": {
    "3pl_id": "3pl456"
  }
}

response = requests.post(url, json=payload)

print("Status:", response.status_code)
print("Result:", response.json())