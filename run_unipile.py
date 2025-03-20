

import requests
import cv2
import os
import numpy as np
import argparse
from genderize import Genderize  # Make sure to install this package: pip install Genderize

# Base URL for the Unipile API
BASE_URL = "https://api10.unipile.com:14058/api/v1/users/"

def get_linkedin_profile(identifier, account_id, api_key):
    """
    Calls the Unipile API to fetch the LinkedIn user profile.
    The request URL includes account_id as a query parameter.
    """
    url = f"{BASE_URL}{identifier}?account_id={account_id}"
    headers = {
        "accept": "application/json",
        "X-API-KEY": api_key
    }
    response = requests.get(url, headers=headers)

    if response.status_code == 400:
        print(f"Error: 400 Bad Request - Please ensure the identifier ({identifier}) is correct.")
        return None
    elif response.status_code != 200:
        response.raise_for_status()

    return response.json()

def download_image(image_url, save_path):
    """
    Downloads an image from image_url and saves it to save_path.
    """
    response = requests.get(image_url)
    response.raise_for_status()
    with open(save_path, 'wb') as f:
        f.write(response.content)
    return save_path

# ======== Import functions and variables from detect.py ========
# Ensure that the main detection logic in detect.py is protected by "if __name__ == '__main__'"
from detect import highlightFace, faceNet, ageNet, genderNet, MODEL_MEAN_VALUES, ageList, genderList

def detect_gender_age(image):
    """
    Detects faces in the image and predicts gender and age for each face.
    Returns a list of dictionaries, each containing gender, gender_confidence, and age.
    """
    padding = 20
    resultImg, faceBoxes = highlightFace(faceNet, image)
    if not faceBoxes:
        print("No face detected.")
        return None

    results = []
    for faceBox in faceBoxes:
        face = image[
            max(0, faceBox[1]-padding) : min(faceBox[3]+padding, image.shape[0]-1),
            max(0, faceBox[0]-padding) : min(faceBox[2]+padding, image.shape[1]-1)
        ]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        # Gender prediction and confidence
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        genderIdx = genderPreds[0].argmax()
        detected_gender = genderList[genderIdx]
        gender_confidence = genderPreds[0][genderIdx]

        # Age prediction (without confidence)
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        results.append({
            "gender": detected_gender,
            "gender_confidence": gender_confidence,
            "age": age
        })
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Fetch LinkedIn user profile via Unipile API and perform gender and age detection."
    )
    parser.add_argument("identifier", help="LinkedIn user's name or id")
    parser.add_argument("account_id", help="Your Unipile account ID")
    parser.add_argument("api_key", help="Your Unipile API key")
    args = parser.parse_args()

    # Fetch the user profile via Unipile API
    profile = get_linkedin_profile(args.identifier, args.account_id, args.api_key)
    if not profile:
        print("Failed to fetch user profile.")
        return

    # Extract first_name and predict gender using the Genderize package
    first_name = profile.get("first_name")
    if first_name:
        genderize_result = Genderize().get([first_name])
        if genderize_result and len(genderize_result) > 0:
            genderize_pred = genderize_result[0]
            genderize_gender = genderize_pred.get("gender")
            genderize_probability = genderize_pred.get("probability")
        else:
            genderize_gender = None
            genderize_probability = None
    else:
        genderize_gender = None
        genderize_probability = None

    # Output Genderize prediction results
    print("Genderize prediction results:")
    print(f"  First Name: {first_name}")
    if genderize_gender:
        print(f"  Gender: {genderize_gender}")
        print(f"  Probability: {genderize_probability}")
    else:
        print("  Unable to predict gender from first name.")

    # Retrieve the image URL from the profile
    image_url = profile.get("profile_picture_url_large")
    if not image_url:
        print("Image URL not found in the response.")
        return

    # Download the image to the current folder, naming it as identifier.jpg
    save_filename = f"{args.identifier}.jpg"
    save_path = os.path.join(os.getcwd(), save_filename)
    print(f"Downloading image to {save_path} ...")
    try:
        download_image(image_url, save_path)
    except Exception as e:
        print("Failed to download image:", e)
        return

    # Read the downloaded image
    image = cv2.imread(save_path)
    if image is None:
        print("Unable to read the downloaded image.")
        return

    # Perform gender and age detection on the image
    results = detect_gender_age(image)
    if not results:
        print("Detection failed or no face detected.")
    else:
        for idx, res in enumerate(results):
            print(f"Face {idx+1} detection results:")
            print(f"  Gender: {res['gender']}")
            print(f"  Gender Confidence: {res['gender_confidence']:.2f}")
            print(f"  Age: {res['age']}")

if __name__ == "__main__":
    main()
