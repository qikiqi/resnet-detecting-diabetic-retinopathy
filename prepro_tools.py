import pandas as pd
import os
import cv2
import numpy as np
from shutil import copyfile
from scipy import ndimage

def remove_not_found(csv_path, location='../train_prepro/', prefix='600_600_300_'):
    data_info = pd.read_csv(csv_path)
    counter = 0
    
    assert 'image' in data_info, "Didn't find 'image' column from the csv found in the path {}".format(csv_path)
    assert os.path.isfile(csv_path), "Didn't find anything from the path {}".format(csv_path)

    not_found = set()
    num_files = len(data_info['image'])
    for item in data_info['image']:
        found_filename = os.path.join(location, prefix + item + '.jpeg')
        if not os.path.isfile(found_filename):
            not_found.add(item)

    counter_not_found = len(not_found)
    print("Skimmed through {} number of files, with {} of them not being in the path {}.".format(num_files, counter_not_found, location))

    with open(csv_path + '.mod', 'w') as outfile, open(csv_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            # If current line (image/(header)) is not a subset of the not_found set,
            # this will yield empty set (False) and that line will be included
            # to the new csv file.
            line_parsed = line.rstrip('\n')[:-2]
            if set([line_parsed]).issubset(not_found):
                continue
            outfile.write(line)
            counter += 1
    
    new_filename = csv_path + '.mod'
    print("Wrote file {}, with {} lines.".format(new_filename, counter))
    
    
def crop_images(output_folder, input_folder='../train_prepro', x=600, y=600):
    
    # Check if input image folder exists
    assert os.path.isdir(input_folder), "Didn't find input folder from {}".format(input_folder)
    
    # Create the folder for output images
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder, exist_ok=False)
        print("The given target folder {} does not exists, creating".format(output_folder))
        
    # Get all the files in a directory
    files_in_dir = os.listdir(input_folder)
    
    assert len(files_in_dir), "Empty folder given as input ({}), exiting...".format(input_folder)
    
    for filename in files_in_dir:
        
        # Set the output filename
        output_filename = str(x) + "_" + str(y) + "_" + filename
        # Set the output folder and join it with the filename
        output_location = os.path.join(output_folder, output_filename)
        
        # If the file doesn't end with .jpeg, do nothing
        if not filename.endswith('.jpeg'):
            continue
        
        # Don't do anything if the file already exists
        if os.path.isfile(output_location):
            #print("File exists, skipping file: {}".format(output_location))
            continue
        try:
            # Open the file with opencv
            input_location = os.path.join(input_folder, filename)
            img_pre = cv2.imread(input_location)
            
            # Get the shape
            height, width, channels = img_pre.shape
            center = (int(height/2), int(width/2))
            
            # Image is desired size
            if height == y and width == x:
                pass
            # Image is too big from x and y
            elif height > y and width > x:
                img_pre = _crop_xy(img_pre, x, y, center)
            # Image is too big from y and too small from x
            # OR
            # Image is too big from y and ok from x
            elif (height > y and width < x) or \
                 (height > y and width == x):
                img_pre = _crop_xy(img_pre, width, y, center)
                # Image is too big from y (fixed above) and too small from x
                if height > y and width < x:
                    img_pre = _fill_img(img_pre, x, height, height, width)
            # Image is too small from y and too big from x
            # OR
            # Image is ok from y and too big from x
            elif (height < y and width > x) or \
                 (height == y and width > x):
                img_pre = _crop_xy(img_pre, x, height, center)
                # Image is too small from y and too big from x (fixed above)
                if height < y and width > x:
                    img_pre = _fill_img(img_pre, width, y, height, width)
            # Image is too small from y and ok from x
            # OR
            # Image is ok from y and too small from x
            # OR
            # Image is too small from x and y
            elif (height < y and width == x) or \
                 (height == y and width < x) or \
                 (height < y and width < x):
                img_pre = _fill_img(img_pre, x, y, height, width)
            

            # Write the image
            cv2.imwrite(output_location, img_pre)
        except Exception as e:
            print("Error in the file: {}".format(filename))
            print(repr(e))
            

def rotate_not_zero(csv_path, input_folder='../train_prepro', prefix='600_600_300_', multiply=5, base_rotation=45, exclude=[0]):
    data_info = pd.read_csv(csv_path)
    
    # Mandatory csv checks
    assert 'image' in data_info, "Didn't find 'image' column from the csv found in the path {}".format(csv_path)
    assert os.path.isfile(csv_path), "Didn't find anything from the path {}".format(csv_path)
    assert base_rotation - (multiply*3) > 0, "Trying to rotate back (or over) to zero {} - ({} * 3) <= 0".format(base_rotation, multiply)
    assert exclude in np.unique(data_info['level']), "One of the labels to " \
            "exclude ({}) can't be found from the data, labels found from data: {}".format(exclude, np.unique(data_info['level']))

    csv_path_out = csv_path + ".rot"
    label_not_zero = []
    for item, label in zip(data_info['image'], data_info['level']):
        if label not in exclude:
            
            # Construct the real filename
            found_filename = os.path.join(input_folder, prefix + item + '.jpeg')
            
            # Skip completely if file not found
            if not os.path.isfile(found_filename):
                print("Can't find file {}".format(found_filename))
                continue
            
            for times in range(multiply):
                info_list = []
                
                # The real filename
                info_list.append(found_filename)
                
                # The amount to rotate
                rotation_num = base_rotation - (times * 3)
                info_list.append(rotation_num)
                
                # The new output filename
                found_filename = os.path.join(input_folder, prefix + item + '_r{}'.format(rotation_num) + '.jpeg')
                info_list.append(found_filename)
                
                # The item name
                info_list.append(item)
                
                # The new item name (append the amount of rotation)
                info_list.append(item + "_r{}".format(rotation_num))
                
                # The label
                info_list.append(label)
                
                # Add the created list to the final list that will be rotated
                #   with defined specs from the list
                label_not_zero.append(info_list)

    for item in label_not_zero:
        
        # Set the full output path
        input_location = item[0]
        
        # Set the full output path
        output_location = item[-4]
        
        # Amount to rotate
        rotation_num = item[1]
        
        # Don't do anything if the file already exists
        if os.path.isfile(output_location):
            print("File exists, skipping file: {}".format(output_location))
            continue
        try:
            
            # Open the file with opencv
            img_pre = cv2.imread(input_location)
            
            # Perform the rotation itself
            img_pre = _rotate_degrees(img_pre, rotation_num)

            # Write the image
            cv2.imwrite(output_location, img_pre)
        except Exception as e:
            print(repr(e))
    
    # Copy the csv_path into new .rot csv_file
    copyfile(csv_path, csv_path_out)
    # And then modify so that the new itemnames are appended to the copied file
    with open(csv_path_out, 'a') as outfile, open(csv_path, 'r') as infile:
        # Determine if the file ends already with a newline
        # (\n, we aint Windows compatible here)
        first_newline = False if "\n" in infile.readlines()[-1] else True
        for item in label_not_zero:
            if first_newline:
                outfile.write("\n")
            outfile.write("{},{}".format(item[-2],item[-1]))
            first_newline = True
    
def _rotate_degrees(img, degrees):
    rotated = ndimage.rotate(img, degrees, reshape=False)
    rotated[np.where((rotated == [0,0,0]).all(axis = 2))] = [128,128,128]
    return rotated

def _crop_xy(img, x, y, center):
    
    try:
        # Calculate the borders from where we crop
        # NOTE: Contains rounding errors
        x2 = x/2
        y2 = y/2
        n_y1 = int(center[0] - y2)
        n_y2 = int(center[0] + y2)
        n_x1 = int(center[1] - x2)
        n_x2 = int(center[1] + x2)
    except TypeError as e:
        print(repr(e))
        print("Exiting")
        quit()
    
    
    img_out = img[n_y1:n_y2, n_x1:n_x2]
    return img_out

# Fill_img doesn't care if the image is already desired size,
#   if (and when) such situation occurs, it will do nothing for that axis
def _fill_img(img, x, y, height, width):
    
    try:
        # Get the color of the grey
        grey_color = (128,128,128)
    
        # Get the needed addition per each side
        # NOTE: Contains rounding errors
        y_border_2_1 = int((y - height) / 2)
        y_border_2_2 = y_border_2_1
        x_border_2_1 = int((x - width) / 2)
        x_border_2_2 = x_border_2_1
        
    except TypeError as e:
        print(repr(e))
        print("Exiting")
        quit()
        
        
    # This manual addition has to be in place since due to rounding errors, we
    #   will receive incorrectly sized images and here we try to mitigate that
    new_height = y_border_2_1 * 2 + height
    if (new_height) != 600:
        if (new_height) == 596:
            y_border_2_1 += 2
            y_border_2_2 += 2
        elif (new_height) == 597:
            y_border_2_1 += 2
            y_border_2_2 += 1 
        elif (new_height) == 598:
            y_border_2_1 += 1
            y_border_2_2 += 1
        elif (new_height) == 599:
            print("New height is 599, adding 2 pixels. This will work on "   \
                  "most pictures, but some will be created with the height " \
                  "of 601, so it is advised to use check_size function from "\
                  "prepro_tools and delete these pictures manually.")
            # This is a rounding error bug that if we add 1 (as we should),
            #   we will still get image of size y-1, this way
            #   we mitigate that problem by creating only few images that are
            #   y+1, and then later fix them by going these through individually
            #   with check_size and remove incorrectly sized images.
            y_border_2_2 += 2
                
    # This manual addition has to be in place since due to rounding errors, we
    #   will receive incorrectly sized images and here we try to mitigate that
    new_width = x_border_2_1 * 2 + width
    if new_width != 600:
        if (new_width) == 596:
            x_border_2_1 += 2
            x_border_2_2 += 2
        elif (new_width) == 597:
            x_border_2_1 += 2
            x_border_2_2 += 1
        elif (new_width) == 598:
            x_border_2_1 += 1
            x_border_2_2 += 1
        elif (new_width) == 599:
            x_border_2_1 += 1 
    
    # Add the borders
    img_out = cv2.copyMakeBorder(
                 img, 
                 y_border_2_1, 
                 y_border_2_2, 
                 x_border_2_1, 
                 x_border_2_2, 
                 cv2.BORDER_CONSTANT, 
                 value=grey_color
              )
        
    return img_out

def check_size(input_folder='../train_prepro', x=600, y=600):
    
    # Check if input image folder exists
    assert os.path.isdir(input_folder), "Didn't find input folder from {}".format(input_folder)
        
    # Get all the files in a directory
    files_in_dir = os.listdir(input_folder)
    
    assert len(files_in_dir), "Empty folder given as input ({}), exiting...".format(input_folder)
    
    errors = 0
    for filename in files_in_dir:
        
        # If the file doesn't end with .jpeg, do nothing
        if not filename.endswith('.jpeg'):
            continue
        
        try:
            # Open the file with opencv
            input_location = os.path.join(input_folder, filename)
            img_pre = cv2.imread(input_location)
            
            # Get the shape
            height, width, channels = img_pre.shape
            
            if height != y:
                print("File {} has height of {}".format(filename, height))
            if width != x:
                print("File {} has width of {}".format(filename, width))
                
            if height != y or width != x:
                errors += 1
            
        except Exception as e:
            print(repr(e))
            
    if errors == 0:
        print("No errors, all images are of the size {}x{}".format(x,y))
    else:
        print("Found {] errors!".format(errors))

def scale_radius(img, scale):
    x_axis = int(img.shape[0] / 2)
    x = img[x_axis,:,:].sum(1)
    r = (x>x.mean()/10).sum()/2
    s = scale * 1.0/r
    return cv2.resize(img,(0,0), fx=s, fy=s)

def preprocess_images(input_folder_name, output_folder_name, scale=300):
    files_in_dir = os.listdir(input_folder_name)
    for filename in files_in_dir:
        # Don't try to preprocess if no .jpeg suffix
        if not filename.endswith('.jpeg'):
            continue
        out_full = os.path.join(output_folder_name, str(scale) + "_" + filename)
        # Don't preprocess if file exists
        if os.path.isfile(out_full):
            continue
        try:
            # Open the file with opencv
            img_pre = cv2.imread(os.path.join(input_folder_name, filename))

            # Scale the image to the radius defined above
            img_pre = scale_radius(img_pre, scale)

            # Define a gaussian blur
            gblur = cv2.GaussianBlur(img_pre, (0,0), scale/30)

            # Substract the local mean color
            img_pre = cv2.addWeighted(img_pre,
                                     4,
                                     gblur,
                                     -4,
                                     128)

            ## Remove outer 10% from the image
            # (Part of the black area that holds no data)
            temp = np.zeros(img_pre.shape)
            center = (int(img_pre.shape[1]/2), int(img_pre.shape[0]/2))
            radius = int(scale*0.9)
            color = (1,1,1)
            thickness = -1
            cv2.circle(temp,
                      center,
                      radius,
                      color,
                      thickness)
            
            # Finalize the preprocessing
            img_pre = img_pre * temp + 128 * (1 - temp)

            # Write the preprocessed image itself
            cv2.imwrite(out_full, img_pre)
        except Exception as e:
            print(repr(e))
