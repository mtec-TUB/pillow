
import os
import re

def find_missing_files(edf_folder, xml_folder):
    """
    Finds missing files by comparing .edf and .xml filenames based on their numbers.
    """
    # Regex to extract the number from the filename
    pattern = re.compile(r'-(aa\d{4})')

    # Get a list of all .edf and .xml files
    edf_files = {os.path.basename(f) for f in os.listdir(edf_folder) if f.endswith('.edf')}
    xml_files = {os.path.basename(f) for f in os.listdir(xml_folder) if f.endswith('.xml')}

    # Extract the unique identifiers (aaXXXX) for comparison
    edf_ids = {pattern.search(f).group(1) for f in edf_files if pattern.search(f)}
    xml_ids = {pattern.search(f).group(1) for f in xml_files if pattern.search(f)}

    # Find the IDs that are unique to each set
    missing_xml_ids = edf_ids - xml_ids
    missing_edf_ids = xml_ids - edf_ids

    # Find the corresponding full filenames for the missing IDs
    missing_xml_files = []
    for missing_id in missing_xml_ids:
        for f in edf_files:
            if missing_id in f:
                # Assuming the pattern mros_visit[0-5]-aaXXXX.edf
                missing_xml_files.append(f.replace('mros_visit', 'mros-visit').replace('.edf', '-nsrr.xml'))

    missing_edf_files = []
    for missing_id in missing_edf_ids:
        for f in xml_files:
            if missing_id in f:
                # Assuming the pattern mros-visit[0-5]-aaXXXX-nsrr.xml
                missing_edf_files.append(f.replace('mros-visit', 'mros_visit').replace('-nsrr.xml', '.edf'))

    return missing_edf_files, missing_xml_files

if __name__ == '__main__':
    # Replace these with your actual folder paths
    edf_folder_path = '/media/linda/Elements/sleep_data/MROS - MrOS Sleep Study/polysomnography/edfs/visit1'
    xml_folder_path = '/media/linda/Elements/sleep_data/MROS - MrOS Sleep Study/polysomnography/annotations-events-nsrr/visit1'

    # Check if the folders exist
    if not os.path.isdir(edf_folder_path):
        print(f"Error: The folder '{edf_folder_path}' does not exist.")
    if not os.path.isdir(xml_folder_path):
        print(f"Error: The folder '{xml_folder_path}' does not exist.")
    if os.path.isdir(edf_folder_path) and os.path.isdir(xml_folder_path):
        missing_edfs, missing_xmls = find_missing_files(edf_folder_path, xml_folder_path)

        print("--- Missing .edf Files ---")
        if missing_edfs:
            for f in sorted(missing_edfs):
                print(f)
        else:
            print("No .edf files are missing.")

        print("\n--- Missing .xml Files ---")
        if missing_xmls:
            for f in sorted(missing_xmls):
                print(f)
        else:
            print("No .xml files are missing.")
