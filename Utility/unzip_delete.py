import os 
import sys 
import timeit 
import zipfile 


def unzip_delete(folder_path, remove=False):
    # check if folder exists 
    if not os.path.isdir(folder_path):
        print("Error: Folder not found!")
        exit(1)
    
    # navigate to the folder 
    os.chdir(folder_path)

    # unzip all zip files in the folder 
    zip_files = [file for file in os.listdir() if file.endswith(".zip")]
    for zip_file in zip_files: 
        with zipfile.ZipFile(zip_file, 'r') as zip_ref: 
            zip_ref.extractall()

    # remove zip files 
    if remove: 
        for zip_file in zip_files: 
            os.remove(zip_file)
        print("Unzipping and file deletion complete!")
        return 

    print("Unzipping complete!")
    return 


def main():  
    if len(sys.argv) < 2 or len(sys.argv) > 3: 
        print("Usage: python3 unzip_delete.py folder_path [delete]")
        sys.exit(1)

    folder_path = sys.argv[1]
    remove = False
    if len(sys.argv) > 2 and sys.argv[2].lower() == 'remove':
        remove = True

    unzip_delete(folder_path, remove)
    

if __name__ == "__main__": 
    time_taken = timeit.timeit(main)
    print(f"Time taken: {time_taken}")
