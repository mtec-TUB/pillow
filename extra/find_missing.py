
import os
import re
import glob
import shutil

def find_missing_files(psg_folder, psg_ext, annot_folder, annot_ext):
    """
    Finds missing files by comparing psg and annot filenames based on their numbers.
    """
    # Regex to extract the number from the filename
    # pattern = re.compile(r'-(aa\d{4})')

    # Get a list of all files
    psg_files = {f for f in glob.glob(psg_folder+'/**/*'+psg_ext, recursive=True)}
    annot_files = {f for f in glob.glob(annot_folder+'/**/*'+annot_ext, recursive=True)}

    # Extract the unique identifiers (aaXXXX) for comparison
    # psg_ids = {pattern.search(f).group(1) for f in psg_files if pattern.search(f)}
    # annot_ids = {pattern.search(f).group(1) for f in annot_files if pattern.search(f)}

    psg_ids = {os.path.basename(f).split('.')[0] for f in psg_files}
    annot_ids = {os.path.basename(f).split('.')[0] for f in annot_files}

    # Find the IDs that are unique to each set
    # missing_annot_ids = psg_ids - annot_ids
    # missing_psg_ids = annot_ids - psg_ids

    missing_annot_file_names = list(psg_ids - annot_ids)
    missing_psg_file_names = list(annot_ids - psg_ids)

    missing_annot_files = []
    lonely_psg_files = []
    for annot_file in missing_annot_file_names:
        missing_annot_files.append(annot_file+annot_ext)
        for f in psg_files:
            if annot_file in f:
                lonely_psg_files.append(f)

    missing_psg_files = []
    lonely_annot_files = []
    for psg_file in missing_psg_file_names:
        missing_psg_files.append(psg_file+psg_ext)
        for f in annot_files:
            if psg_file in f:
                lonely_annot_files.append(f)

    # # Find the corresponding full filenames for the missing IDs
    # missing_annot_files = []
    # for missing_id in missing_annot_ids:
    #     for f in psg_files:
    #         if missing_id in f:
    #             # Assuming the pattern mros_visit[0-5]-aaXXXX.edf
    #             missing_annot_files.append(f.replace('mros_visit', 'mros-visit').replace(psg_ext, '-nsrr.xml'))

    # missing_psg_files = []
    # for missing_id in missing_psg_ids:
    #     for f in annot_files:
    #         if missing_id in f:
    #             # Assuming the pattern mros-visit[0-5]-aaXXXX-nsrr.xml
    #             missing_psg_files.append(f.replace('mros-visit', 'mros_visit').replace('-nsrr.xml', psg_ext))

    return missing_psg_files, missing_annot_files, lonely_psg_files, lonely_annot_files

if __name__ == '__main__':
    # Replace these with your actual folder paths
    psg_folder_path = '/media/linda/Elements/sleep_data/MROS - MrOS Sleep Study/polysomnography/edfs/visit1'
    annot_folder_path = '/media/linda/Elements/sleep_data/MROS - MrOS Sleep Study/polysomnography/annotations-events-nsrr/visit1'

    psg_folder_path = '/media/linda/Elements/sleep_data/STAGES - Stanford Technology Analytics and Genomics in Sleep/original/STAGES PSGs'
    annot_folder_path = '/media/linda/Elements/sleep_data/STAGES - Stanford Technology Analytics and Genomics in Sleep/original/STAGES PSGs'

    output_dir = '/media/linda/Elements/sleep_data/STAGES - Stanford Technology Analytics and Genomics in Sleep/original/missing/'

    psg_folder_path = '/home/linda/Downloads/MNC - Mignot Nature Communications'
    annot_folder_path = '/home/linda/Downloads/MNC - Mignot Nature Communications'

    output_dir = '/home/linda/Downloads/MNC - Mignot Nature Communications/missing'

    # Check if the folders exist
    if not os.path.isdir(psg_folder_path):
        print(f"Error: The folder '{psg_folder_path}' does not exist.")
    if not os.path.isdir(annot_folder_path):
        print(f"Error: The folder '{annot_folder_path}' does not exist.")
    if os.path.isdir(psg_folder_path) and os.path.isdir(annot_folder_path):
        missing_psgs, missing_annots, lonely_psg, lonely_annot = find_missing_files(psg_folder_path, '.edf', annot_folder_path, '.xml')

        print(lonely_annot)
        print(lonely_psg)

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        print("--- Missing psg Files ---")
        if missing_psgs:
            for f in sorted(missing_psgs):
                print(f)
            for f in lonely_annot:
                shutil.move(f, output_dir)
        else:
            print("No psg files are missing.")

        print("\n--- Missing annot Files ---")
        if missing_annots:
            for f in sorted(missing_annots):
                print(f)
            for f in lonely_psg:
                shutil.move(f, output_dir)
        else:
            print("No annot files are missing.")
