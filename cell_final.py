import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_cell_location(mcc: int, mnc: int, lac: int, cid: int, radio: str = "lte"):
    """
    Simple function to get cell tower location using Unwired Labs API.

    Args:
        mcc (int): Mobile Country Code (e.g., 404 for India)
        mnc (int): Mobile Network Code (e.g., 45 for Jio)
        lac (int): Location Area Code (LTE/5G: Tracking Area Code)
        cid (int): Cell ID
        radio (str): Radio type (lte, gsm, umts, nr)

    Returns:
        dict: Contains latitude, longitude, accuracy, and address
    """

    API_URL = "https://us1.unwiredlabs.com/v2/process.php"
    API_KEY = os.getenv("UNWIRED_API_KEY")  # set your token via env var

    if not API_KEY:
        print("❌ Please set your UNWIRED_API_KEY environment variable.")
        return None

    payload = {
        "token": API_KEY,
        "radio": radio,
        "mcc": mcc,
        "mnc": mnc,
        "cells": [{"lac": lac, "cid": cid}],
        "address": 1
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=10)
        data = response.json()

        if data.get("status") != "ok":
            print("⚠️ API Error:", data.get("message", "Unknown error"))
            return None

        return {
            "latitude": data.get("lat"),
            "longitude": data.get("lon"),
            "accuracy_m": data.get("accuracy"),
            "address": data.get("address"),
        }

    except Exception as e:
        print("❌ Error:", e)
        return None


# Example usage
if __name__ == "__main__":
    result = get_cell_location(404, 45, 1234, 5678901)
    if result:
        print("\n📍 Location Info:")
        for k, v in result.items():
            print(f"{k}: {v}")
