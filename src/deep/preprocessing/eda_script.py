import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random


CUTOFF = 1e-6   # matches the constant used throughout the project
ROOT_DIR = os.path.join("data", "training")


# Load the raw metadata folder
def load_metadata(root_dir):
    metadata = pd.read_csv(os.path.join(root_dir, "meta_data.csv"))
    metadata["end"] = pd.to_datetime(metadata["end"])
    metadata = metadata.sort_values(by="end").reset_index(drop=True)
    metadata["is_flare"] = metadata["peak_flux"] > CUTOFF
    metadata["region"] = metadata["id"].apply(lambda d: d.split("_", 1)[0])
    return metadata


def main():
    metadata = load_metadata(ROOT_DIR)
    n = len(metadata)
    print(f"Total training samples: {n}")
    print(f"Unique active regions: {metadata['region'].nunique()}")
    print()
    

    log_flux = np.log10(metadata["peak_flux"].clip(lower=1e-12))
    plt.hist(log_flux, bins=60, color="steelblue", edgecolor="none")
    plt.axvline(np.log10(CUTOFF), color="red", linestyle="--",
                     label=f"CUTOFF = {CUTOFF:.0e}")
    for boundary, label in [(-7, "A"), (-6, "B"), (-5, "C"), (-4, "M")]:
        plt.axvline(boundary, color="gray", linestyle=":", alpha=0.6)
        plt.text(boundary, plt.ylim()[1]*0.95, label, ha='center', fontsize=10)
    plt.xlabel("log10(peak_flux)  [W/m^2]")
    plt.ylabel("Count")
    plt.title("Distribution of Peak Flux")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Find the most powerful flare ---
    flares = metadata[metadata["is_flare"] == True]
    strongest_flare_idx = metadata["peak_flux"].idxmax()
    strongest_flare = metadata.loc[strongest_flare_idx]
    
    print("\n" + "="*70)
    print("STRONGEST FLARE IN DATASET")
    print("="*70)
    print(f"ID:           {strongest_flare['id']}")
    print(f"Region:       {strongest_flare['region']}")
    print(f"End time:     {strongest_flare['end']}")
    print(f"Peak flux:    {strongest_flare['peak_flux']:.3e} W/m²")
    print(f"Log10 flux:   {np.log10(strongest_flare['peak_flux']):.2f}")
    print(f"Flare class:  {get_flare_class(strongest_flare['peak_flux'])}")
    print()

    # --- Find a random non-flare ---
    non_flares = metadata[metadata["is_flare"] == False]
    random_non_flare_idx = random.choice(non_flares.index)
    random_non_flare = metadata.loc[random_non_flare_idx]
    
    print("="*70)
    print("RANDOM NON-FLARE FROM DATASET")
    print("="*70)
    print(f"ID:           {random_non_flare['id']}")
    print(f"Region:       {random_non_flare['region']}")
    print(f"End time:     {random_non_flare['end']}")
    print(f"Peak flux:    {random_non_flare['peak_flux']:.3e} W/m²")
    print(f"Log10 flux:   {np.log10(random_non_flare['peak_flux']):.2f}")
    print(f"Flare class:  Below detection threshold")
    print()

    # --- Summary statistics ---
    print("="*70)
    print("DATASET STATISTICS")
    print("="*70)
    print(f"Total samples:     {n}")
    print(f"Flares:            {len(flares)} ({100*len(flares)/n:.1f}%)")
    print(f"Non-flares:        {len(non_flares)} ({100*len(non_flares)/n:.1f}%)")
    print(f"Strongest flare:   {strongest_flare['peak_flux']:.3e} W/m²")
    print(f"Weakest flare:     {flares['peak_flux'].min():.3e} W/m²")
    print(f"Max non-flare:     {non_flares['peak_flux'].max():.3e} W/m²")


def get_flare_class(peak_flux):
    """Convert peak flux to GOES flare classification"""
    if peak_flux < 1e-8:
        return "Below A-class"
    elif peak_flux < 1e-7:
        return "A-class"
    elif peak_flux < 1e-6:
        return "B-class"
    elif peak_flux < 1e-5:
        return "C-class"
    elif peak_flux < 1e-4:
        return "M-class"
    else:
        return "X-class"


if __name__ == "__main__":
    main()