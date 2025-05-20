
import pandas as pd
import os
from PIL import Image, ImageDraw, ImageChops
import numpy as np
from scipy import stats


class convertColors:

    def lms2RGB(lms_Nx3, conversionMatrix=None):
        """Convert from cone space (Long, Medium, Short) to RGB.

        Requires a conversion matrix, which will be generated from generic
        Sony Trinitron phosphors if not supplied (note that you will not get
        an accurate representation of the color space unless you supply a
        conversion matrix)

        usage::

            rgb_Nx3 = lms2rgb(dkl_Nx3(el,az,radius), conversionMatrix)

        """

        # its easier to use in the other orientation!
        lms_3xN = np.transpose(lms_Nx3)

        if conversionMatrix is None:
            # cones_to_rgb = np.asarray([ # default from psychopy
            #     # L        M        S
            #     [4.97068857, -4.14354132, 0.17285275],  # R
            #     [-0.90913894, 2.15671326, -0.24757432],  # G
            #     [-0.03976551, -0.14253782, 1.18230333]])  # B
            
            cones_to_rgb = np.asarray([ # from https://ixora.io/projects/colorblindness/color-blindness-simulation-research/
                # L        M        S
                [5.47221206, -4.6419601, 0.16963708],  # R
                [-1.1252419, 2.29317094, -0.1678952],  # G
                [0.02980165, -0.19318073, 1.16364789]])  # B

            # print('This monitor has not been color-calibrated. Using default LMS conversion matrix.')
        else:
            cones_to_rgb = conversionMatrix

        RGB = np.dot(cones_to_rgb, lms_3xN)
        return np.transpose(RGB)  # return in the shape we received it

    def XYZtoRGB(XYZ_Nx3):
        '''
        Coverts XYZ to RGB
        Args:
            XYZ_Nx3: A numpy array with [X, Y, Z]
        Returns:
            RGB: A numpy array with [R, G, B]
        '''
        M = np.asarray([
            [ 2.0413690, -0.5649464, -0.3446944],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0134474, -0.1183897, 1.0154096]])
        # # Increase luminance (increase Y)
        # XYZ_Nx3 = np.array([XYZ_Nx3[0],XYZ_Nx3[1]+0.01,XYZ_Nx3[2]])
        # [M]^-1 * [X; Y; Z] = [R; G; B]
        RGB = np.dot(M, XYZ_Nx3)
        return np.transpose(RGB)

 
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
   
    # df['check'] = np.array([stats.percentileofscore(df[pred_col], x, kind='weak') for x in df[pred_col]])
    # df['check'] = np.log10(df[pred_col].values+0.001)
    # df = df[df[pred_col] > df[pred_col].quantile(0.1)]
    # df = df[df[pred_col] < df[pred_col].quantile(0.9)]
    # df[pred_col] = df[pred_col] - min(df[pred_col])
    # df[pred_col] = df[pred_col] / max(df[pred_col])

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
        rows[pred_col] = rows[pred_col].rank(pct=True)
       
        for _, row in rows.iterrows():
            y, x = int(row['X']), int(row['Y'])  # Switched x and y

            # Plot the ratios from blue to red
            Ratio = row[pred_col]

            if not np.isnan(Ratio):
                # Determine color based on 'correct' column with more transparency
                color = (int(Ratio*255), 0, int((1-Ratio)*255), 255) # Red if more HbO2, Blue if more Hb
                # Highlight only the specific pixel with transparency
                draw.point((x, y), fill=color)
                

            # color = convertColors.lms2RGB(np.array([row['L'],row['M'],row['S']]))
            # color = convertColors.XYZtoRGB(np.array([row['cmfX'], row['cmfY'], row['cmfZ']]))

            # color = XYZtoRGB(np.array([row['cmfX'], row['cmfY'], row['cmfZ']]))
            # color = tuple(int(3*num) for num in color)
            # Highlight only the specific pixel with transparency
            draw.point((x, y), fill=color)
       
        # Blend the highlighted image with the original using alpha compositing
        blended_image = Image.alpha_composite(image, image_copy)
       
        # Save the blended image
        output_path = os.path.join(output_folder, f"{float_name}_highlighted.png")
        blended_image.save(output_path)
        print(f"Saved blended image: {output_path}")
    print('Done saving images')

 
predLocsPath = "/Users/maycaj/Documents/HSI_III/5-19-25_Round1.csv"
RGBfolder = "/Volumes/lsr-conway/PROJECTS/23_ Hyperspectral_Imaging/HSI Data/Round 1 and 2/RGB"
pred_col = 'Hb/HbO2'
LabelledOutputFolder = '/Users/maycaj/Documents/HSI_III/HbO2 Preds 416 and 931nm'

highlight_pixels(predLocsPath, RGBfolder, LabelledOutputFolder, pred_col)



pass