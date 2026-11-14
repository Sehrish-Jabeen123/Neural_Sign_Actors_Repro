import urllib.request

def download_sample_data():
    print("Internet connection test kiya ja raha hai...")
    
    # Yeh PyTorch ka official aur hamesha chalne wala link hai
    url = "https://raw.githubusercontent.com/pytorch/pytorch/main/README.md"
    file_name = "test_readme.md"
    
    try:
        urllib.request.urlretrieve(url, file_name)
        print(f"Success! File '{file_name}' download ho gayi.")
        print("Aap ka server internet se data download kar sakta hai.")
    except Exception as e:
        print("Error aya:", e)

if __name__ == "__main__":
    download_sample_data()