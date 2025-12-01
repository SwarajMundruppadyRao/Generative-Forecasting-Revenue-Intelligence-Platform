import requests
import json
import sys

def test_endpoints():
    base_url = "http://localhost:8000"
    
    print("="*50)
    print("🧪 TESTING API ENDPOINTS")
    print("="*50)
    
    # 1. Test Forecast
    print("\n1. 📊 Testing Forecast Endpoint...")
    payload = {
        "store_id": 1,
        "dept_id": 1,
        "horizon": 4,
        "model_type": "lstm",
        "natural_language_query": "Explain the forecast"
    }
    print(f"Request: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(f"{base_url}/forecast", json=payload)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("\n✅ Response:")
            print(f"Predictions: {data['predictions']}")
            print(f"Dates: {data['forecast_dates']}")
            if data.get('explanation'):
                print(f"Explanation: {data['explanation']}")
        else:
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        print("Make sure the API server is running!")

    # 2. Test RAG
    print("\n" + "-"*50)
    print("\n2. 🤖 Testing RAG Question Answering...")
    payload = {
        "question": "Which stores are similar to Store 1?",
        "store_id": 1
    }
    print(f"Question: {payload['question']}")
    
    try:
        response = requests.post(f"{base_url}/rag-answer", json=payload)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("\n✅ Answer:")
            print(data['answer'])
            if data.get('sources'):
                print(f"\nSources: {len(data['sources'])} documents found")
        else:
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Connection Error: {e}")

if __name__ == "__main__":
    test_endpoints()
