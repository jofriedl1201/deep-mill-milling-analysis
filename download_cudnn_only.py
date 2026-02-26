import urllib.request
import os
import sys

# Official Redist URL for cuDNN 8.9.7 for CUDA 12.x (Windows)
URL = "https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-8.9.7.29_cuda12-archive.zip"
FILENAME = "cudnn_8.9.7_cuda12.zip"

def download():
    print(f"Target URL: {URL}")
    print(f"Destination: {os.path.abspath(FILENAME)}")
    
    if os.path.exists(FILENAME):
        size = os.path.getsize(FILENAME)
        print(f"File already exists. Size: {size/1024/1024:.2f} MB")
        if size > 100 * 1024 * 1024: # Expecting ~600MB+ for cuDNN
             print("File size looks reasonable.")
             return
        else:
             print("File looks incomplete. Re-downloading...")
    
    print("Downloading...")
    try:
        def reporthook(blocknum, blocksize, totalsize):
            readsoFar = blocknum * blocksize
            if totalsize > 0:
                percent = readsoFar * 1e2 / totalsize
                if int(percent) % 10 == 0:
                    sys.stdout.write(f"\r{percent:.0f}%")
        
        urllib.request.urlretrieve(URL, FILENAME, reporthook)
        print("\nDownload complete.")
        size = os.path.getsize(FILENAME)
        print(f"Final Size: {size/1024/1024:.2f} MB")
    except Exception as e:
        print(f"Download failed: {e}")

if __name__ == "__main__":
    download()
