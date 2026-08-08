"""
This class is a basic script that handles model preprocessing, testing, and training
"""

def preprocess_data():
    print("Preprocessing Images for CNN Training...")

def main():
    print("="*40)
    print("SOLAR FLARE PREDICTIOR")
    print("by Ankith Seethesh Vaidya and Matthew Geisel")
    print("\nThis project uses the SDOBenchmark taken from the SDO satellite")
    print("\nOptions: \n1. Preprocess Data\n2. Train Models\n3. Test Models\n4. Model Demonstration")
    selection = input("\nChoose an Option: ")
    print()

    if selection == "1":
        preprocess_data()



if __name__ == "__main__":
    main()