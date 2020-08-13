import pandas as pd


def main():
    # Read csv file
    pokemon = pd.read_csv("data/pokemon.csv")
    x = pokemon[["Type 1", "Type 2"]].copy()
    y = pokemon["Legendary"].copy()

    # Check first column
    print("Hello")


if __name__ == "__main__":
    main()
