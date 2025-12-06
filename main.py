from ml_project.data import load_raw_data, clean_data

def main():
    print(load_raw_data())
    print(clean_data())

if __name__ == "__main__":
    main()