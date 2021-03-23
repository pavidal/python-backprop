import argparse as ap
import numpy as np
import pandas as pd


class Node:
    def __init__(self, weights):
        self.delta_w = np.full(len(weights), 0.0)
        self.bias = weights[-1]
        self.weights = weights[:-1]
        self.output = None

    def out(self, data):
        output = self.bias

        # Sum of node's weights
        for k in range(len(self.weights)):
            # print(self.weights[k], data[k], self.weights[k] * data[k])
            output += self.weights[k] * data[k]

        self.output = 1 / (1 + np.exp(-output))

        return self.output

    # Unused
    def error(self, data, predictand):
        output = 0

        # Sum of node's weights
        for k in range(len(self.weights)):
            output += self.weights[k] * data[k]

        return predictand - output

    def adjust_weights(self, delta, data, step):
        new_bias = self.bias + step * delta + args.momentum * self.delta_w[-1]
        self.delta_w[-1] = new_bias - self.bias
        self.bias = new_bias

        for j in range(len(self.weights)):
            new_weight = self.weights[j] + step * delta * data[j] + args.momentum * self.delta_w[j]
            self.delta_w[j] = new_weight - self.weights[j]
            self.weights[j] = new_weight


def generate_weights(n):
    node_weights = []

    for w in range(n):
        # Weights of all predictors in a node
        # [0, 1)
        node_weights.append(np.random.uniform(0, 1))

    return node_weights


# Data standardisation functions


def standardise_minmax(dataframe, limit):
    """
    Standardise data within the dataframe to a range

    :param dataframe: A Pandas dataframe containing data not normalised
    :param limit: Boolean to specify [0.1, 0.9] limit
    :return: A Pandas dataframe containing normalised data
    :rtype: pandas.DataFrame
    """

    data = dataframe.copy()

    for col in data.columns:
        if col == data.columns[-1]:
            preprocess_values.update({
                "min": data[col].min(),
                "max": data[col].max()
            })

        # standardise data to [0, 1]
        data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())

        # Limit range to [0.1, 0.9]
        if limit:
            data[col] = 0.8 * data[col] + 0.1

    return data


def standardise_square(dataframe):
    """
    Standardise data within the dataframe with respect to the sum of squares of all values

    :param dataframe: A Pandas dataframe containing data not normalised
    :return: A Pandas dataframe containing normalised data
    :rtype: pandas.DataFrame
    """

    data = dataframe.copy()

    for col in data.columns:
        if col == data.columns[-1]:
            preprocess_values.update({
                "square": ((data[col] ** 2).sum()) ** (1 / 2)
            })

        data[col] = data[col] / ((data[col] ** 2).sum()) ** (1 / 2)

    return data


def standardise_stddev(dataframe):
    """
        Standardise data within the dataframe with respect to the mean and standard deviation of the data

        :param dataframe: A Pandas dataframe containing data not normalised
        :return: A Pandas dataframe containing normalised data
        :rtype: pandas.DataFrame
    """

    data = dataframe.copy()

    for col in data.columns:
        if col == data.columns[-1]:
            preprocess_values.update({
                "stddev": data[col].std(),
                "mean": data[col].mean()
            })

        data[col] = (data[col] - data[col].mean()) / data[col].std()

    return data


def destandardise_square(array):
    data = []

    square = preprocess_values.get("square")

    for r in array:
        data.append(square * r)

    return pd.DataFrame(data)


def destandardise_minmax(array, limit):
    data = []

    min = preprocess_values.get("min")
    max = preprocess_values.get("max")

    for r in array:
        x = r
        if limit:
            x = (r - 0.1) / 0.8

        data.append((x * (max - min)) + min)

    return pd.DataFrame(data)


def destandardise_stddev(array):
    data = []

    stddev = preprocess_values.get("stddev")
    mean = preprocess_values.get("mean")

    for r in array:
        data.append((r * stddev) + mean)

    return pd.DataFrame(data)

# Back Propagation


def back_propagation(df, nodes, step, epochs):
    for epoch in range(epochs):
        print(epoch + 1, "of", epochs)
        d = 0

        for row in df.to_numpy():
            predictand = row[-1]
            data = row[:-1]

            fp = forward_pass(data, nodes)
            uo = fp.get("uo")
            hidden_out = fp.get("hidden")

            if np.isnan(uo) or np.isinf(uo):
                raise RuntimeError("Float Overflow")

            # Backward Pass
            delta_o = (predictand - uo) * (uo * (1 - uo))
            d = delta_o     # for logging error in each

            # Updating weights
            output_node.adjust_weights(delta=delta_o, step=step, data=hidden_out)

            # hidden nodes
            for n in range(len(nodes)):
                node = nodes[n]
                delta_j = output_node.weights[n] * delta_o * (node.output * (1 - node.output))
                node.adjust_weights(delta=delta_j, step=step, data=data)

        error_list.append(np.absolute(d))


def forward_pass(data, nodes):
    hidden_out = []

    # forward pass
    for node in nodes:
        output = node.out(data=data)
        hidden_out.append(output)

    # Output Node
    uo = output_node.out(data=hidden_out)

    return {"uo": uo, "hidden": hidden_out}


# I/O functions


def get_data(path):
    """
    Gets data from a CSV or Excel file and outputs a pandas DataFrame.
    It assumes that the data being fed is correct; GIGO.

    :param path: Path to datasheet
    :return: A Pandas dataframe
    :rtype: pandas.DataFrame
    """

    # Check for file type
    if path[-3:] == "csv":
        dataset = pd.read_csv(path)
    elif path[-4:] == "xlsx" or path[-3:] == "xls":
        dataset = pd.read_excel(path)
    else:
        raise NotImplementedError("This file type is not supported. This script only accepts *.csv and *.xlsx files.")

    # Cleanup dataset
    dataset = dataset.dropna(how="all", axis=1)  # Remove empty columns
    dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]  # Remove unnamed columns

    return dataset


def store_weights(n):
    nw = []
    for node in n:
        nw.append(node.weights)

    # Store as CSV
    pd.DataFrame(nw).to_csv(args.data + ".weights.csv", index=False)


if __name__ == '__main__':

    # create an argument parser for CLI
    parser = ap.ArgumentParser()
    parser.add_argument("-d", "--data", help="path to datasheet", metavar="<PATH>", type=str, default="data.csv")
    parser.add_argument("-s", "--standard", help="type of data standardisation, available options: "
                                                 "<none|minmax|minmax90|square|stddev>",
                        metavar="<OPTION>", type=str, default="none")
    parser.add_argument("-e", "--epochs", help="number of epochs to train on this data (default = 5,000)",
                        metavar="<EPOCHS>", type=int, default=5000)
    parser.add_argument("-n", "--nodes", help="number of nodes to train on this data (default = 10)", metavar="<NODES>",
                        type=int, default=5)
    parser.add_argument("-p", "--step", help=" (default = 0.01)", metavar="<STEP>",
                        type=float, default=0.01)
    parser.add_argument("-m", "--momentum", help=" (default = 0.0)", metavar="<MOMENTUM>",
                        type=float, default=0.0)
    # parser.add_argument("-p", "--predictand", help="column to use as result, -1 = auto", metavar="<COLUMN>", type=int,
    #                     default=-1)

    # TODO: Refactor into an init() function

    args = parser.parse_args()

    error_list = []

    try:
        raw_data = get_data(args.data)

        preprocess_values = {}

        # Pre-processing data
        std_methods = {
            "none": raw_data.copy(),
            "minmax": standardise_minmax(raw_data, False),
            "minmax90": standardise_minmax(raw_data, True),
            "square": standardise_square(raw_data),
            "stddev": standardise_stddev(raw_data)
        }

        if args.standard not in std_methods:
            print("Standardisation argument is invalid.")
            exit()

        preprocessed = std_methods.get(args.standard, "none")  # used to derive the partitions
        preprocessed.to_csv(args.data+".preprocessed.csv", index=False)

        # Partitioning data
        array_len = len(preprocessed)
        training_set = preprocessed.head(int(array_len * 0.6))
        validation_set = preprocessed.tail(int(array_len * 0.4))  # used to derive the last two
        testing_set = validation_set.head(int(len(validation_set / 2)))
        unseen_set = validation_set.tail(int(len(validation_set / 2)))

        # Initial weights
        nodes = []
        for i in range(args.nodes):
            nodes.append(Node(weights=generate_weights(len(preprocessed.columns))))

        output_node = Node(weights=generate_weights(args.nodes + 1))

        # Train a model

        back_propagation(training_set, nodes, args.step, args.epochs)
        pd.DataFrame(error_list).to_csv(args.data + ".error.csv", index=False)

        # TODO: Add backprop improvements

        # Validate Model

        results = []

        for row in validation_set.to_numpy():
            predictand = row[-1]
            data = row[:-1]

            fp = forward_pass(data=data, nodes=nodes)
            uo = fp.get("uo")

            delta_o = (predictand - uo) * (uo * (1 - uo))
            results.append(uo)

        reverse_std = {
            "none": pd.DataFrame(results),
            "minmax": destandardise_minmax(results, False),
            "minmax90": destandardise_minmax(results, True),
            "square": destandardise_square(results),
            "stddev": destandardise_stddev(results)
        }

        prediction = reverse_std.get(args.standard)
        prediction.columns = ["Prediction"]

        real_values = raw_data.tail(int(array_len * 0.4)).loc[:, [preprocessed.columns[-1]]]

        results = pd.concat([real_values, prediction], axis=1)
        results.to_csv(args.data + ".validate.csv", index=False)

        # TODO: Test model on unseen data

        nodes.append(output_node)
        store_weights(nodes)

    except FileNotFoundError as e:
        print("File does not exist.")
        exit()
    except NotImplementedError as e:
        print("File is an unsupported format. This script only accepts *.csv and *.xlsx files.")
        exit()
    except RuntimeError as e:
        print("Integer overflow detected")
        pd.DataFrame(error_list).to_csv(args.data + ".error.csv", index=False)
        exit()
