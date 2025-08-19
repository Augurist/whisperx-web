#!/usr/bin/env python3
import requests
import sys

BASE_URL = "http://localhost:5000"

def test_endpoint(name, url, expected_status=200):
    try:
        response = requests.get(f"{BASE_URL}{url}")
        if response.status_code == expected_status:
            print(f"✅ {name}: OK")
            return True
        else:
            print(f"❌ {name}: Got {response.status_code}, expected {expected_status}")
            return False
    except Exception as e:
        print(f"❌ {name}: {e}")
        return False

print("Testing WhisperX deployment...")
print("-" * 40)

tests = [
    ("Home page", "/"),
    ("Speakers page", "/speakers"),
    ("Admin page", "/admin"),
    ("Health check", "/health"),
    ("API: Speakers list", "/api/speakers"),
    ("API: Speaker clips", "/api/all_speaker_clips"),
    ("API: Speaker texts", "/api/speaker_texts"),
    ("API: Check updates", "/api/check_updates"),
]

results = []
for name, url in tests:
    results.append(test_endpoint(name, url))

print("-" * 40)
if all(results):
    print("✅ All tests passed!")
    sys.exit(0)
else:
    print("❌ Some tests failed")
    sys.exit(1)
