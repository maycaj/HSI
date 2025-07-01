### Plot the predictions on the RGB images with color coding based on the ratio of HbO2 to Hb or using other chromophores.
# How to use: Adjust pred_col to be the column name of the predictions in the CSV and set the other paths accordingly.

from sklearn.preprocessing import QuantileTransformer
import pandas as pd
import os
from PIL import Image, ImageDraw, ImageChops
import numpy as np
from scipy import stats
from SpectrumClassifier2 import ChromDotSpectra
import pyarrow.csv as pv

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

    print('Loading CSV...')
    table = pv.read_csv(csv_path)
    df = table.to_pandas()
    print('Done Loading CSV')
    os.makedirs(output_folder, exist_ok=True)
   
   # Select specific images
    df = df[df['FloatName'].isin(['Edema 12 image 3 float','Edema 36 Image 3 float','Edema 19 Image 1 float'])]

    # Get unique images
    unique_images = df['FloatName'].unique()
   
    image_files = {f.lower(): f for f in os.listdir(image_folder)}
   
    for float_name in unique_images:
        image_filename = f"{float_name}.png".lower()
        if image_filename not in image_files:
            print(f"Image {float_name}.png not found. Exiting...")
            break
        
        df_sel = df[df['FloatName'] == float_name] # Selected dataframe
       
        image_path = os.path.join(image_folder, image_files[image_filename])
       
        # Open image and create a copy
        image = Image.open(image_path).convert("RGBA")
        image_copy = Image.new("RGBA", image.size, (0, 0, 0, 0))  # Create a transparent overlay
        draw = ImageDraw.Draw(image_copy, "RGBA")
        
        # Hb and HbO2 from LLS
        # df_sel[pred_col] = df_sel['yInterp HbO2 cm-1/M'] / (df_sel['yInterp Hb cm-1/M'] + df_sel['yInterp HbO2 cm-1/M'])

        # # # Dot product with hemoglobins 
        HbO2Path = '/Users/maycaj/Documents/HSI/Absorbances/HbO2 Absorbance.csv'
        nmStart = 'Wavelength_451.18' #'Wavelength_411.27' #'Wavelength_411.27'
        nmEnd = 'Wavelength_954.83' #'Wavelength_987.82' #'Wavelength_1004.39'
        ChromDotSpectra.chrom_dot_spectra(df_sel, [nmStart,nmEnd], 'HbO2 cm-1/M', 'HbO2', HbO2Path, plot=False) # Find HbO2 for each pixel
        ChromDotSpectra.chrom_dot_spectra(df_sel, [nmStart,nmEnd], 'Hb cm-1/M', 'Hb', HbO2Path, plot=False) # Find Hb for each pixel
        df_sel[pred_col] = df_sel['HbO2'] / df_sel['Hb']
        # sel_columns = [pred_col,'X','Y']
        # df_sel = df_sel[sel_columns]

        ## Normalize within each image
        # df_sel[pred_col] = df_sel[pred_col].rank(pct=True)
        # df_sel['check'] = np.array([stats.percentileofscore(df[pred_col], x, kind='weak') for x in df[pred_col]])
        # df_sel[pred_col] = np.log10(df_sel[pred_col].values+0.001)
        # qt = QuantileTransformer(output_distribution='normal', random_state=0)
        # df_sel['check'] = qt.fit_transform(df[[pred_col]])
        # df_sel = df_sel[df_sel[pred_col] != 0]
        # df_sel = df_sel[df_sel[pred_col] < df_sel[pred_col].quantile(0.999)]
        # df_sel = df_sel[df_sel[pred_col] > df_sel[pred_col].quantile(0.001)]
        df_sel[pred_col] = df_sel[pred_col] - min(df_sel[pred_col])
        df_sel[pred_col] = df_sel[pred_col] / max(df_sel[pred_col])

       
        for _, row in df_sel.iterrows():
            y, x = int(row['X']), int(row['Y'])  # X and Y are switched in the .csv

            # Plot the ratios from blue to red
            Ratio = row[pred_col]

            if not np.isnan(Ratio):
                # Determine color based on 'correct' column with more transparency
                color = (int(Ratio*255), int(Ratio*255), int(Ratio*255), 255) # Red if more HbO2, Blue if more Hb
                # Highlight only the specific pixel with transparency
                draw.point((x, y), fill=color)

            # color = convertColors.lms2RGB(np.array([row['L'],row['M'],row['S']]))
            # color = convertColors.XYZtoRGB(np.array([row['cmfX'], row['cmfY'], row['cmfZ']]))

            # color = XYZtoRGB(np.array([row['cmfX'], row['cmfY'], row['cmfZ']]))
            # color = tuple(int(3*num) for num in color)
            # Highlight only the specific pixel with transparency
            # draw.point((x, y), fill=color)
       
        # Blend the highlighted image with the original using alpha compositing
        blended_image = Image.alpha_composite(image, image_copy)
       
        # Save the blended image
        output_path = os.path.join(output_folder, f"{float_name}_highlighted.png")
        blended_image.save(output_path)
        print(f"Saved blended image: {output_path}")
    print('Done saving images')

if __name__ == '__main__':
    predLocsPath = '/Users/maycaj/Documents/HSI/PatchCSVs/June_9_NOCRAndSmoothing_FullRound1and2AllWLs.csv' # Path to the CSV with predictions and locations
    RGBfolder = "/Users/maycaj/Documents/HSI/RGB" # RGB images folder
    pred_col = 'HbO2/Hb' # Column name of the predictions in the CSV
    LabelledOutputFolder = '/Users/maycaj/Documents/HSI/Model Preds/HbO2:Hb Jun 9' # Output folder for the highlighted images
    highlight_pixels(predLocsPath, RGBfolder, LabelledOutputFolder, pred_col)
