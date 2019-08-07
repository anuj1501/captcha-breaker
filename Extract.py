import os
import os.path
import cv2
import glob

Captcha_Folder = 'generated_captcha_images'
Extract_Folder = 'extracted_letter_images'

# Get a list of all captch images we need to process
captcha_image_files = glob.glob(os.path.join(Captcha_Folder, "*"))
counts = {}

for (i, captch_image_file) in enumerate(captcha_image_files):
    print("Status:: Processing Image {}/{}".format(i+1, len(captcha_image_files)))


    # Filename is the captcha text
    filename = os.path.basename(captch_image_file)
    captcha_correct_text = os.path.splitext(filename)[0]

    # Load image and manipulate
    image = cv2.imread(captch_image_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # find the contours (continuous blobs of pixels)
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0]
    
    letter_regions = []
    
    for contour in contours:
        # Get rectangle that contains the contour
        (x, y, w, h) = cv2.boundingRect(contour)
        # Compare the width and height of the contour to detect letters
        if w/h > 1.25:
            # This contour is too wide to have a single letters
            half_width = int(w/2)
            letter_regions.append((x, y, half_width, h))
            letter_regions.append((x+half_width, y, half_width, h))
        else:
            # Normal image with one letter
            letter_regions.append((x, y, w, h))

    # if less than 4 letters.Skip the image instead of saving bad data
    if len(letter_regions) != 4:
        continue

    # Sort the detected letter images based on x coordinate 
    letter_regions = sorted(letter_regions, key=lambda x: x[0])

    for letter_bounding_box, letter_text in zip(letter_regions, captcha_correct_text):
        x, y, w, h = letter_bounding_box
        # Extract letter from original image with 2 pixel margin around edge
        letter_image = gray[y-2:y+h+2, x-2:x+w+2]

        # Save image in folder
        save_path = os.path.join(Extract_Folder, letter_text)

        if not os.path.exists(save_path):
            os.makedirs(save_path) 
        # Write letters to a file
        count = counts.get(letter_text, 1)
        p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
        cv2.imwrite(p, letter_image)

        # increment the count for the current key
        counts[letter_text] = count+1
