
import pandas as pd
import os
from PIL import Image, ImageDraw, ImageChops
import numpy as np
 
def highlight_pixels(csv_path, image_folder, output_folder, pred_col):
    '''
    Takes RGB image and labels each pixel a color, and saves to a new folder
    Args:
        csv_path: filepath of csv with predictions and locations 
        image_folder: filepath of where the unedited RGB photos are
        output_folder: where the highlighted RGB images will be saved
        csv_col: the column name of the predictions
    '''


    df = pd.read_csv(csv_path)
    os.makedirs(output_folder, exist_ok=True)

    # convert relevant column to percentiles
    df = df[df[pred_col] > df[pred_col].quantile(0.01)]
    df = df[df[pred_col] < df[pred_col].quantile(0.99)]
   
    # Get unique images
    unique_images = df['FloatName'].unique()
   
 
    image_files = {f.lower(): f for f in os.listdir(image_folder)}
   
    for float_name in unique_images:
        image_filename = f"{float_name}.png".lower()
        if image_filename not in image_files:
            print(f"Image {float_name}.png not found. Exiting...")
            break
       
        image_path = os.path.join(image_folder, image_files[image_filename])
       
        # Open image and create a copy
        image = Image.open(image_path).convert("RGBA")
        image_copy = Image.new("RGBA", image.size, (0, 0, 0, 0))  # Create a transparent overlay
        draw = ImageDraw.Draw(image_copy, "RGBA")
       
 
        rows = df[df['FloatName'] == float_name]
       
        for _, row in rows.iterrows():
            y, x = int(row['X']), int(row['Y'])  # Switched x and y
            Ratio = row[pred_col]

            if not np.isnan(Ratio):
                # Determine color based on 'correct' column with more transparency
                color = (int(Ratio*255), 0, int((1-Ratio)*255), 100) # Red if more HbO2, Blue if more Hb
            
                # Highlight only the specific pixel with transparency
                draw.point((x, y), fill=color)
       
        # Blend the highlighted image with the original using alpha compositing
        blended_image = Image.alpha_composite(image, image_copy)
       
        # Save the blended image
        output_path = os.path.join(output_folder, f"{float_name}_highlighted.png")
        blended_image.save(output_path)
        print(f"Saved blended image: {output_path}")
    print('Done saving images')
 
 
csv_path = "/Users/maycaj/Documents/HSI_III/1D Classifier/HbpredLocsMar10_Noarm.csv"
# Local Hemoglobin Paths
# image_folder = "/Users/maycaj/Documents/HSI_III/RGB"
# output_folder = "/Users/maycaj/Documents/HSI_III/ModelPredsHb"

image_folder = "/Volumes/lsr-conway/PROJECTS/23_ Hyperspectral_Imaging/HSI Data/Patients Enrollments  Nov 25/RGB"

# # Hemoglobin Paths
# output_folder = "/Volumes/lsr-conway/PROJECTS/23_ Hyperspectral_Imaging/HSI Data/Patients Enrollments  Nov 25/ModelPredsHb/Mar 20 2025"
# pred_col = 'HbO2/Hb'

# Pheomelanin Paths
output_folder = "/Volumes/lsr-conway/PROJECTS/23_ Hyperspectral_Imaging/HSI Data/Patients Enrollments  Nov 25/ModelPredsPheomel"
pred_col = 'Pheomelanin'

highlight_pixels(csv_path, image_folder, output_folder, pred_col)
