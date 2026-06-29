import requests
import json

BASE_URL = 'http://localhost:5000'

print("="*80)
print("AIRBNB PRICE PREDICTION API - TEST SUITE")
print("="*80)

# ============================================================================
# TEST 1: Health Check
# ============================================================================
print("\n[TEST 1] Testing /health endpoint...")
try:
    response = requests.get(f'{BASE_URL}/health')
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    if response.status_code == 200:
        print("✓ PASS - API is running")
    else:
        print("✗ FAIL - Unexpected status code")
except Exception as e:
    print(f"✗ FAIL - Error: {e}")

# ============================================================================
# TEST 2: Info Endpoint
# ============================================================================
print("\n" + "="*80)
print("[TEST 2] Testing /info endpoint...")
try:
    response = requests.get(f'{BASE_URL}/info')
    print(f"Status Code: {response.status_code}")
    info_data = response.json()
    print(f"Model Type: {info_data['model_info']['type']}")
    print(f"R² Score: {info_data['model_info']['r2_score']}")
    print(f"RMSE: {info_data['model_info']['rmse']}")
    print(f"Available Neighbourhoods: {len(info_data['categorical_features']['Neighbourhood'])}")
    print(f"Room Types: {info_data['categorical_features']['Room Type']}")
    
    if response.status_code == 200:
        print("✓ PASS - Info endpoint working")
    else:
        print("✗ FAIL - Unexpected status code")
except Exception as e:
    print(f"✗ FAIL - Error: {e}")

# ============================================================================
# TEST 3: Prediction - Valid Input
# ============================================================================
print("\n" + "="*80)
print("[TEST 3] Testing /predict endpoint with VALID input...")
try:
    prediction_data = {
        "Number of Reviews": 50,
        "Reviews per Month": 2.5,
        "Minimum Nights": 30,
        "Availability 365": 200,
        "Beds": 1,
        "Years_as_Host": 3.5,
        "Neighbourhood": "Manhattan",
        "Room Type": "Entire home/apt",
        "Neighbourhood_RoomType": "Manhattan_Entire home/apt"
    }
    
    response = requests.post(
        f'{BASE_URL}/predict',
        json=prediction_data
    )
    
    print(f"Status Code: {response.status_code}")
    result = response.json()
    
    print(f"\nPrediction Result:")
    print(f"  Predicted Price: ${result['predicted_price']}")
    print(f"  Confidence Range: ${result['confidence_interval']['low']} - ${result['confidence_interval']['high']}")
    print(f"  RMSE: ${result['confidence_interval']['rmse']}")
    
    if response.status_code == 200 and result['status'] == 'success':
        print("\n✓ PASS - Prediction successful")
    else:
        print("\n✗ FAIL - Unexpected response")
except Exception as e:
    print(f"✗ FAIL - Error: {e}")

# ============================================================================
# TEST 4: Prediction - Missing Data
# ============================================================================
print("\n" + "="*80)
print("[TEST 4] Testing /predict with MISSING data (should fail)...")
try:
    incomplete_data = {
        "Number of Reviews": 50,
        "Neighbourhood": "Manhattan"
        # Missing other required fields
    }
    
    response = requests.post(
        f'{BASE_URL}/predict',
        json=incomplete_data
    )
    
    print(f"Status Code: {response.status_code}")
    error_result = response.json()
    print(f"Error: {error_result.get('error')}")
    print(f"Message: {error_result.get('message')}")
    
    if response.status_code == 400:
        print("\n✓ PASS - Correctly rejected incomplete data")
    else:
        print("\n✗ FAIL - Should have returned 400 error")
except Exception as e:
    print(f"✗ FAIL - Error: {e}")

# ============================================================================
# TEST 5: Prediction - Invalid Category
# ============================================================================
print("\n" + "="*80)
print("[TEST 5] Testing /predict with INVALID category (should fail)...")
try:
    invalid_data = {
        "Number of Reviews": 50,
        "Reviews per Month": 2.5,
        "Minimum Nights": 30,
        "Availability 365": 200,
        "Beds": 1,
        "Years_as_Host": 3.5,
        "Neighbourhood": "InvalidNeighbourhood",  # Invalid
        "Room Type": "Entire home/apt",
        "Neighbourhood_RoomType": "InvalidNeighbourhood_Entire home/apt"
    }
    
    response = requests.post(
        f'{BASE_URL}/predict',
        json=invalid_data
    )
    
    print(f"Status Code: {response.status_code}")
    error_result = response.json()
    print(f"Error: {error_result.get('error')}")
    print(f"Message: {error_result.get('message')}")
    
    if response.status_code == 400:
        print("\n✓ PASS - Correctly rejected invalid category")
    else:
        print("\n✗ FAIL - Should have returned 400 error")
except Exception as e:
    print(f"✗ FAIL - Error: {e}")

# ============================================================================
# TEST 6: Multiple Predictions
# ============================================================================
print("\n" + "="*80)
print("[TEST 6] Testing multiple predictions...")

test_cases = [
    {
        "name": "Budget Room",
        "data": {
            "Number of Reviews": 20,
            "Reviews per Month": 1.0,
            "Minimum Nights": 60,
            "Availability 365": 100,
            "Beds": 1,
            "Years_as_Host": 1.0,
            "Neighbourhood": "Queens",
            "Room Type": "Shared room",
            "Neighbourhood_RoomType": "Queens_Shared room"
        }
    },
    {
        "name": "Premium Apartment",
        "data": {
            "Number of Reviews": 100,
            "Reviews per Month": 5.0,
            "Minimum Nights": 2,
            "Availability 365": 300,
            "Beds": 3,
            "Years_as_Host": 5.0,
            "Neighbourhood": "Manhattan",
            "Room Type": "Entire home/apt",
            "Neighbourhood_RoomType": "Manhattan_Entire home/apt"
        }
    },
    {
        "name": "Mid-Range Room",
        "data": {
            "Number of Reviews": 60,
            "Reviews per Month": 3.0,
            "Minimum Nights": 7,
            "Availability 365": 250,
            "Beds": 2,
            "Years_as_Host": 2.5,
            "Neighbourhood": "Brooklyn",
            "Room Type": "Private room",
            "Neighbourhood_RoomType": "Brooklyn_Private room"
        }
    }
]

for i, test_case in enumerate(test_cases, 1):
    try:
        response = requests.post(
            f'{BASE_URL}/predict',
            json=test_case['data']
        )
        result = response.json()
        
        print(f"\n  {i}. {test_case['name']}")
        print(f"     Predicted Price: ${result['predicted_price']}")
        print(f"     Range: ${result['confidence_interval']['low']} - ${result['confidence_interval']['high']}")
        print(f"     ✓ Success")
    except Exception as e:
        print(f"\n  {i}. {test_case['name']}")
        print(f"     ✗ Error: {e}")

# ============================================================================
# TEST SUMMARY
# ============================================================================
print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)
print("""
✓ Health check - Verify API is running
✓ Info endpoint - Get model and feature information
✓ Valid prediction - Test with complete data
✓ Missing data - Test error handling
✓ Invalid category - Test validation
✓ Multiple predictions - Test various scenarios

All tests completed!
""")