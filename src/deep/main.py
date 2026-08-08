from training.train_model import train_model
from evaluation.evaluate_model import evaluate_model
from preprocessing.flare_dataset import FlareDataset
VALID_SELECTIONS = ["1", "2", "3", "4"]
FEATURES = ["magnetogram", "continuum", "304", "94"]

if __name__ == "__main__":
    print("Solar Flare Prediction: Deep Learning")
    print("This uses CNNs to predict whether solar flares will occur")


    while( True ):
        print("\nOptions:")
        print("1. Magnetogram")
        print("2. Continuum")
        print("3. 304 A")
        print("4. 94 A")
        print("Anything else to exit")
        model_selection = input("Model to evaluate:")
        if model_selection not in VALID_SELECTIONS:
            break

        print("\nModel options:")
        print("1. Train")
        print("2. Test")
        action_selection = input("Action: ")

        if action_selection == "1":
            train_model(FEATURES[int(model_selection)-1])
        else:
            evaluate_model(FEATURES[int(model_selection)-1])
