import os
import shutil

def main():
    source_folder_path = './PNGS/train'
    destination_folder_path = './PNGS/test'
    move(source_folder_path, destination_folder_path)

def move(source_folder, destination_folder):

    files = os.listdir(source_folder)

    # Iterate through files in the source folder
    for file_name in files:
        # Check if the file is an image (you can extend this list as needed)
        # Construct paths for the source and destination images
        source_path = os.path.join(source_folder, file_name)
        print(file_name[-9::])
        if (file_name[-9::] == "1_RGB.png"):
            destination_folder = destination_folder
            destination_path = os.path.join(destination_folder, file_name)
            
        # Move the image file to the destination folder
            shutil.move(source_path, destination_path)
            print(f"Moved '{file_name}' to '{destination_folder}'.")


main()