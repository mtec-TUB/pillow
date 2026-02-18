import os
import shutil
import pyedflib


def fix_edf_date_mismatch_low_level(filepath):
    """
    Fixes an EDF file by directly manipulating the header's binary data
    to correct a date mismatch.

    Args:
        filepath (str): The path to the EDF or BDF file.
    """
    try:
        # The C++ code indicates that the file's main header is 256 bytes.
        # The start date field is at byte offset 168 and is 8 bytes long (dd.mm.yy).
        # The local recording identification field is at byte offset 88 and is 80 bytes long.

        with open(filepath, 'r+b') as f:
            # Read the first 256 bytes of the header
            header_bytes = bytearray(f.read(256))

            # Extract the 80-byte local recording identification field (offsets 88-167)
            local_recording_id_bytes = header_bytes[88:168].decode('ascii', errors='ignore')

            # Find the date string in the local recording ID (e.g., "19-SEP-1994")
            # This logic is based on the analysis of the C++ code.
            correct_date_str = None
            for part in local_recording_id_bytes.split():
                if len(part) == 11 and part[2] == '-' and part[6] == '-':
                    try:
                        day_str = part[0:2]
                        month_str = part[3:6].upper()
                        year_str = part[7:11]
                        
                        # We need to convert the month from 'MMM' format to a number
                        month_map = {'JAN': '01', 'FEB': '02', 'MAR': '03', 'APR': '04', 'MAY': '05', 'JUN': '06',
                                     'JUL': '07', 'AUG': '08', 'SEP': '09', 'OCT': '10', 'NOV': '11', 'DEC': '12'}
                        month_num = month_map.get(month_str)

                        if month_num:
                            correct_date_str = f"{day_str}.{month_num}.{year_str[2:]}"
                            break
                    except (ValueError, IndexError):
                        continue

            if correct_date_str:
                # The corrected date must be exactly 8 characters long to fit the header
                if len(correct_date_str) == 8:
                    # Convert the new date string to bytes
                    new_date_bytes = correct_date_str.encode('ascii')

                    # Overwrite the old date field at byte offset 168 with the new date
                    f.seek(168, 0) # Go to the start date offset
                    f.write(new_date_bytes)
                    print(f"File '{filepath}' has been successfully repaired.")
                else:
                    print(f"Error: The corrected date string '{correct_date_str}' is not 8 characters long. Cannot fix.")
            else:
                print(f"Could not find a parsable date in the local recording ID for '{filepath}'. No action taken.")

    except Exception as e:
        print(f"An error occurred while processing '{filepath}': {e}")


def sanitize_ascii(byte_string):
    """
    Forces a byte string into 7-bit ASCII compatibility by replacing
    non-ASCII characters with a placeholder ('?'). This mimics the
    C++ code's sanitization logic of the EdfBrowser Header Editor.
    """
    # Decode using 'latin-1' which won't error on any byte value.
    # Re-encode to 'ascii', replacing any character that doesn't fit.
    return byte_string.decode('latin-1').encode('ascii', 'replace')

def low_level_repair_edf(file_path, output_dir, broken_output_dir):
    """
    Performs a low-level byte repair on a corrupted EDF/BDF header,
    including sanitizing all text fields to be 7-bit ASCII compliant.
    """
    try:
        pyedflib.EdfReader(file_path)
        # print(f'{file_path}: repairing not necessary')
        shutil.copy(file_path,output_dir)
        return
    except:
        try:
            with open(file_path, 'rb') as f:
                main_header = f.read(256)
                if len(main_header) < 256:
                    print(f"  ERROR: File '{os.path.basename(file_path)}' is too small.")
                    return
        
                ns = int(main_header[252:256].decode('ascii').strip())
                total_header_size = (ns + 1) * 256
        
                f.seek(0)
                raw_header = f.read(total_header_size)
                raw_data = f.read()
        
        except (ValueError, IOError) as e:
            print(f"  CRITICAL ERROR reading '{os.path.basename(file_path)}': {e}.")
            return
    
    header_array = bytearray(raw_header)
    
    # --- Sanitize Main Header Text Fields ---
    header_array[8:88] = sanitize_ascii(header_array[8:88]) # Patient Info
    header_array[88:168] = sanitize_ascii(header_array[88:168]) # Recording Info
    print("Sanitized main header text fields.")
    
    # --- Fix Starttime ---
    if b'NA' in header_array[168:184]:
        print(f'current starttime: {header_array[168:184]}')
        print(f'starttime: {header_array[168:184]} changed to 01.01.8500.00.00')
        header_array[168:176] = b'01.01.85'
        header_array[176:184] = b'00.00.00'
    
    
    if header_array[244:252] == b'0       ':
        header_array[244:252] = b'1       '
        print("  Fix: Corrected datarecord duration from 0 to 1.")

    idx_marker = header_array.find('EDF+C'.encode())
    reserved_field = header_array[idx_marker:idx_marker+5]
    if reserved_field:
        print(f" Fix: Detected '{reserved_field.strip()}' marker. Clearing reserved field to demote from EDF+.")
        header_array[idx_marker:idx_marker+5] = b' ' * len(reserved_field)
    
    is_bdf = raw_header[0] == 255
    
    # --- Sanitize and Fix Each Signal Header ---
    for i in range(ns):
        # Define all field offsets for the current signal 'i'
        label_offset = 256 + (i * 16)
        transducer_offset = 256 + (ns * 16) + (i * 80)
        physdim_offset = 256 + (ns * 96) + (i * 8)
        prefilter_offset = 256 + (ns * 136) + (i * 80)
        dmin_offset = 256 + (ns * 120) + (i * 8)
        dmax_offset = 256 + (ns * 128) + (i * 8)
        pmin_offset = 256 + (ns * 104) + (i * 8)
        pmax_offset = 256 + (ns * 112) + (i * 8)
        reserved_offset = 256 + (ns * 224) + (i * 32)

        label_bytes = header_array[label_offset:label_offset+16]
        label = label_bytes.decode('ascii', errors='ignore').strip()
    
        # Sanitize all text fields for this signal
        header_array[label_offset:label_offset+16] = sanitize_ascii(header_array[label_offset:label_offset+16])
        header_array[transducer_offset:transducer_offset+80] = sanitize_ascii(header_array[transducer_offset:transducer_offset+80])
        header_array[physdim_offset:physdim_offset+8] = sanitize_ascii(header_array[physdim_offset:physdim_offset+8])
        header_array[prefilter_offset:prefilter_offset+80] = sanitize_ascii(header_array[prefilter_offset:prefilter_offset+80])
        
        header_array[reserved_offset:reserved_offset+32] = b' '*32
    
        try:
            dmin = int(header_array[dmin_offset:dmin_offset+8].decode('ascii').strip())
            dmax = int(header_array[dmax_offset:dmax_offset+8].decode('ascii').strip())

            dmin_ann = -8388608 if is_bdf else -32768
            dmax_ann = 8388607 if is_bdf else 32767
            if dmin < dmin_ann or dmin >= dmax_ann:
                print(f"{label} Before: dmin: {dmin}")
                header_array[dmin_offset:dmin_offset+8] = f"{dmin_ann:<8}".encode('ascii')
                print(f"After: dmin: {int(header_array[dmin_offset:dmin_offset+8].decode('ascii').strip())}")
            if dmax > dmax_ann or dmax <= dmin_ann:
                print(f"{label} Before: dmax: {dmax}")
                header_array[dmax_offset:dmax_offset+8] = f"{dmax_ann:<8}".encode('ascii')
                print(f"After: dmax: {int(header_array[dmax_offset:dmax_offset+8].decode('ascii').strip())}")

            pmin = float(header_array[pmin_offset:pmin_offset+8].decode('ascii').strip())
            pmax = float(header_array[pmax_offset:pmax_offset+8].decode('ascii').strip())
            if pmin >= pmax:
                print(f"{label} Before: pmin: {pmin}, pmax: {pmax}")
                pmin_fixed = -1.0
                pmax_fixed = 1.0
                header_array[pmin_offset:pmin_offset+8] = f"{pmin_fixed:<8}".encode('ascii')
                header_array[pmax_offset:pmax_offset+8] = f"{pmax_fixed:<8}".encode('ascii')
                print(f"After: pmin: {float(header_array[pmin_offset:pmin_offset+8].decode('ascii').strip())}, pmax: {float(header_array[pmax_offset:pmax_offset+8].decode('ascii').strip())}")
    
        except (ValueError, UnicodeDecodeError) as e:
            print(f"  Warning (Signal {i}): Could not parse min/max values after sanitizing. Error: {e}")
            continue

    output_path = os.path.join(output_dir,os.path.basename(file_path))
    with open(output_path, 'wb') as f_out:
        f_out.write(header_array)
        f_out.write(raw_data)

    try:
        pyedflib.EdfReader(output_path)
    except Exception as e:
        print(f'File {file_path} could not be repaired {e}')#, will be moved to broken file folder')
        # if not os.path.exists(broken_output_dir):
        #     os.mkdir(broken_output_dir)
        # os.remove(output_path)
        # shutil.move(file_path,broken_output_dir)
        return


    print(f"  Repaired file saved successfully")
    return


def batch_repair_directory(source_dir, output_dir=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files_to_process = [f for f in os.listdir(source_dir) if f.lower().endswith(('.edf', '.bdf'))]
    if not files_to_process:
        print(f"No .edf or .bdf files found in '{source_dir}'.")
        return

    print(f"Found {len(files_to_process)} files to repair.")
    for i, filename in enumerate(files_to_process):
        print(f"Processing file {i}/{len(files_to_process)}: '{filename}'")
        source_path = os.path.join(source_dir, filename)
        if not output_dir:
            repaired_dir = source_dir + '_repaired/'
        else:
            repaired_dir = output_dir

        broken_dir = os.path.join(source_dir + '_broken/')
       
        low_level_repair_edf(source_path, repaired_dir, broken_dir)

    print("\nBatch repair process finished.")


if __name__ == '__main__':

    # source_directory = '/home/linda/Downloads/MNC - Mignot Nature Communications/sleep_data/ssc'
    # output_directory = '/home/linda/Downloads/MNC - Mignot Nature Communications/sleep_data/ssc_repaired'

    source_directory = '/run/user/1004/gvfs/smb-share:server=truenas.local,share=db/db/physio/sleep/NCHSDB - NCH Sleep DataBank/sleep_data'
    output_directory = '/run/user/1004/gvfs/smb-share:server=truenas.local,share=db/db/physio/sleep/NCHSDB - NCH Sleep DataBank/sleep_data_repaired'


    # source_directory = '/media/linda/Volume/NCHSDB - NCH Sleep DataBank/sleep_data_broken'
    # output_directory = '/media/linda/Volume/NCHSDB - NCH Sleep DataBank/sleep_data_repaired'

    # source_directory = '/media/linda/Elements/sleep_data/Sleep-EDFX - Sleep-EDF Expanded/1.0.0/sleep-telemetry'
    # output_directory = '/media/linda/Elements/sleep_data/Sleep-EDFX - Sleep-EDF Expanded/1.0.0/sleep-telemetry_repaired'

    # source_directory = '/run/user/1004/gvfs/smb-share:server=truenas.local,share=db/sleep data/APOE - Sleep Disordered Breathing, ApoE and Lipid Metabolism/original/PSG'
    # output_directory = '/run/user/1004/gvfs/smb-share:server=truenas.local,share=db/sleep data/APOE - Sleep Disordered Breathing, ApoE and Lipid Metabolism/original/PSG_repaired'

    # source_directory = "/media/linda/Elements/sleep_data/CAPSLPDB - CAP Sleep Database/"
    # output_directory = "/media/linda/Elements/sleep_data/CAPSLPDB - CAP Sleep Database/repaired"

    # source_directory = "/run/user/1004/gvfs/smb-share:server=truenas.local,share=db/sleep data/BESTAIR - Best Apnea Interventions in Research/polysomnography/edfs/nonrandomized"
    # output_directory = "/run/user/1004/gvfs/smb-share:server=truenas.local,share=db/sleep data/BESTAIR - Best Apnea Interventions in Research/polysomnography/edfs/nonrandomized_repaired"

    batch_repair_directory(source_directory, output_directory)
    
    # fix_edf_date_mismatch_low_level('/home/linda/Downloads/Sleep-EDFX - Sleep-EDF Expanded/1.0.0/ST7021JM-Hypnogram.edf')
