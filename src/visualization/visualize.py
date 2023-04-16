""" Visualization class for DMD solution """

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PostPlotDMD:
    def __init__(self, dict_output):
        self.data = dict_output

    def plot_singular_values(self, visualization_library="matplotlib"):
        data = self.data["s"][:, np.newaxis]
        data = np.insert(data, 1, range(1, data.shape[0] + 1), axis=1)
        if visualization_library == "matplotlib":
            # Plotting scatter plot using matplotlib
            fig, ax = plt.subplots()
            ax.scatter(data[:, 1], data[:, 0])

            # Setting x and y axis labels and scales
            ax.set_xlabel("i")
            ax.set_ylabel("Singular Values")
            ax.set_yscale("log")

            # Displaying plot
            plt.show()
        elif visualization_library == "seaborn":
            # Plotting scatter plot using seaborn
            sns.scatterplot(x=data[:, 1], y=data[:, 0])

            # Setting x and y axis labels and scales
            plt.xlabel("i")
            plt.ylabel("Singular Values")
            plt.yscale("log")

            # Displaying plot
            plt.show()
        elif visualization_library == "plotly":
            # Plotting scatter plot using plotly
            fig = go.Figure(data=go.Scatter(x=data[:, 1], y=data[:, 0], mode="markers"))

            # Setting x and y axis labels and scales
            fig.update_layout(
                xaxis_title="i", yaxis_title="Singular Values", yaxis_type="log"
            )

            # Displaying plot
            fig.show()
        else:
            logger.log(
                "Invalid visualization library. Please choose 'matplotlib', 'seaborn' or 'plotly'"
            )

    def plot_eigenvalues(self, visualization_library="matplotlib"):
        data = self.data["eigenvals_original"]
        real_part = np.real(data)
        imag_part = np.imag(data)
        if visualization_library == "matplotlib":
            # Plotting scatter plot using matplotlib
            fig, ax = plt.subplots()

            # Plotting the unitary circle
            theta = np.linspace(0, 2 * np.pi, 100)
            ax.plot(np.cos(theta), np.sin(theta), linestyle="--", color="grey")

            # Plotting the real and imaginary parts of eigenvalues
            ax.scatter(real_part, imag_part)

            # Setting x and y axis labels
            ax.set_xlabel("Real")
            ax.set_ylabel("Imaginary")

            # Setting axis limits to -1.5 and 1.5 to fit the unitary circle
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)

            # Displaying plot
            plt.show()
        elif visualization_library == "seaborn":
            # Plotting scatter plot using seaborn
            sns.scatterplot(x=real_part, y=imag_part)

            # Adding unitary circle as a line plot
            theta = np.linspace(0, 2 * np.pi, 1000)
            sns.scatterplot(x=np.cos(theta), y=np.sin(theta), color="grey")

            # Setting x and y axis labels
            plt.xlabel("Real")
            plt.ylabel("Imaginary")

            # Setting axis limits to -1.5 and 1.5 to fit the unitary circle
            plt.xlim(-1.5, 1.5)
            plt.ylim(-1.5, 1.5)

            # Displaying plot
            plt.show()
        elif visualization_library == "plotly":
            # Plotting scatter plot using plotly
            fig = go.Figure(data=go.Scatter(x=real_part, y=imag_part, mode="markers"))

            # Adding unitary circle as a line plot
            theta = np.linspace(0, 2 * np.pi, 100)
            fig.add_trace(
                go.Scatter(
                    x=np.cos(theta),
                    y=np.sin(theta),
                    mode="lines",
                    line=dict(color="grey"),
                )
            )

            # Setting x and y axis labels
            fig.update_layout(xaxis_title="Real", yaxis_title="Imaginary")

            # Setting axis limits to -1.5 and 1.5 to fit the unitary circle
            fig.update_xaxes(range=[-1.5, 1.5])
            fig.update_yaxes(range=[-1.5, 1.5])

            # Displaying plot
            fig.show()
        else:
            logger.log(
                "Invalid visualization library. Please choose 'matplotlib', 'seaborn' or 'plotly'"
            )
