import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def pie_plot(data):
    labels = 'No Legendary', 'Legendary'
    explode = (0.2, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

    fig1, ax1 = plt.subplots()
    ax1.pie(data.value_counts(normalize=True), explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(15, 6)
    plt.savefig("plots/target_pie_chart.png", dpi=100)
    plt.show()


def dist_plot(p_type, pokemon):
    sns.catplot(x=p_type, kind="count", hue="Legendary", data=pokemon)
    plt.title("%s distribution" % p_type)
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(15, 6)
    # plt.savefig("plots/dist_%s.png" % p_type, dpi=100)
    plt.show()


def dist_plot_legendaries(p_type, pokemon):
    pokemon = pokemon[pokemon["Legendary"] == True]
    sns.catplot(x=p_type, kind="count", data=pokemon)
    plt.title("%s distribution" % p_type)
    plt.xticks(rotation='vertical')
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(15, 10)
    plt.savefig("plots/dist_legendaries.png", dpi=100)
    plt.show()


def main():
    # Read csv file
    pokemon = pd.read_csv("data/pokemon.csv")

    # Check first column for na
    print(pokemon["Type 1"].isna().sum())

    # Replace nans in second column - most pokemon don't have secondary type
    pokemon["Type 2"] = pokemon["Type 2"].fillna("no_type")

    # Check if imbalanced dataset (No Legendary: 92%, Legendary: 8%)
    print(pokemon["Legendary"].value_counts(normalize=True))
    pie_plot(pokemon["Legendary"])

    # Plot distribution of Type 1
    dist_plot("Type 1", pokemon)

    # Plot distribution of Type 2
    dist_plot("Type 2", pokemon)

    # Plot Type 1 and Type 2 distributions
    pokemon["combined"] = pokemon["Type 1"] + "-" + pokemon["Type 2"]
    dist_plot_legendaries("combined", pokemon)


if __name__ == "__main__":
    main()
