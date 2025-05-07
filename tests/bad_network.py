import os, urllib.request
def main():
    urllib.request.urlopen("http://malicious.site")   # B310 / MEDIUM
if __name__ == "__main__":
    main()