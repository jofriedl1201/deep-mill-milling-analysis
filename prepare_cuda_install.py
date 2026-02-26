import urllib.request
import os
import sys

INSTALLER_URL = "https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_531.14_windows.exe"
FILENAME = "cuda_12.1_installer.exe"

def download_installer():
    if os.path.exists(FILENAME):
        size = os.path.getsize(FILENAME)
        print(f"File {FILENAME} already exists. Size: {size/1024/1024:.2f} MB")
        # Approx 3GB check
        if size > 2.5 * 1024 * 1024 * 1024:
             print("File seems complete.")
             return
        else:
             print("File seems incomplete. Re-downloading...")
    
    print(f"Downloading {INSTALLER_URL}...")
    try:
        def reporthook(blocknum, blocksize, totalsize):
            readsoFar = blocknum * blocksize
            if totalsize > 0:
                percent = readsoFar * 1e2 / totalsize
                if int(percent) % 10 == 0:
                    sys.stdout.write(f"\r{percent:.0f}%")
        
        urllib.request.urlretrieve(INSTALLER_URL, FILENAME, reporthook)
        print("\nDownload complete.")
        size = os.path.getsize(FILENAME)
        print(f"Downloaded to: {os.path.abspath(FILENAME)}")
        print(f"Size: {size/1024/1024:.2f} MB")
    except Exception as e:
        print(f"Download failed: {e}")

if __name__ == "__main__":
    download_installer()
