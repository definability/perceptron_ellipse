from dataclasses import dataclass, field
from typing import Optional

from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent
from matplotlib.figure import Figure
from matplotlib.pyplot import subplots, show
from matplotlib.widgets import Button, TextBox
from numba import njit
from numpy import (
    array,
    kron,
    linalg,
    linspace,
    meshgrid,
    ndarray,
    zeros,
)
from numpy.linalg import matrix_rank


@dataclass
class Problem:
    inner_points: list = field(default_factory=list)
    outer_points: list = field(default_factory=list)
    alpha: ndarray = field(default_factory=lambda: array((0,) * 6 + (-1,), float))
    is_trained: bool = False


@dataclass
class UI:
    figure: Figure
    axes: Axes
    button_train: Button
    text_steps: TextBox


@njit
def warp_vector(x: ndarray):
    """Convert 2D point to feature vector using Kronecker product."""
    result = zeros(x.size ** 2 + x.size + 1)
    result[:x.size ** 2] = kron(x, x)
    result[x.size ** 2:x.size ** 2 + x.size] = x
    result[-1] = 1
    return result


@njit
def check_point(alpha: ndarray, x: ndarray):
    """Evaluate quadratic form at point x."""
    return alpha.dot(warp_vector(x))


@njit
def fetch_matrix(alpha: ndarray):
    """Convert feature vector to matrix."""
    return alpha[:4].reshape(2, 2)


@njit
def fetch_centre(alpha: ndarray):
    """Calculate centre of the ellipse."""
    return linalg.solve(fetch_matrix(alpha), alpha[4:6] / -2)


@njit
def correct_alpha(alpha: ndarray):
    """Correct alpha to ensure the consistency of the last element."""
    if alpha[-1] == 0:
        return

    matrix = fetch_matrix(alpha)
    if matrix_rank(matrix) < 2:
        return

    centre = fetch_centre(alpha)
    quadratic_form = centre.dot(matrix).dot(centre)

    multiplier = quadratic_form - alpha[-1]
    if multiplier == 0:
        return False

    alpha /= multiplier

    return multiplier <= 0


def train_step(problem: Problem):
    """Perform one training step. Return True if training is complete."""
    for point in problem.inner_points:
        if check_point(problem.alpha, point) >= 0:
            problem.alpha -= warp_vector(point)
            return True

    for point in problem.outer_points:
        if check_point(problem.alpha, point) <= 0:
            problem.alpha += warp_vector(point)
            return True

    matrix = problem.alpha[:4].reshape(2, 2)
    eigenvalues, eigenvectors = linalg.eigh(matrix)
    for eigenvalue, eigenvector in zip(eigenvalues, eigenvectors.T):
        if eigenvalue <= 0:
            problem.alpha[:4] += kron(eigenvector, eigenvector)
            return True

    return correct_alpha(problem.alpha)


@njit
def draw_grid(alpha: ndarray, x, y):
    """Create grid for contour."""
    z = zeros(x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z[i, j] = check_point(alpha, array([x[i, j], y[i, j]]))
    return z


def draw_quadric(event: Optional[MouseEvent], problem: Problem, ui: UI):
    """Visualize the current quadratic form."""
    xlim = ui.axes.get_xlim()
    ylim = ui.axes.get_ylim()
    ui.axes.cla()
    ui.axes.set_xlim(xlim)
    ui.axes.set_ylim(ylim)
    ui.axes.grid(True)

    x, y = meshgrid(linspace(xlim[0], xlim[1], 100),
                    linspace(ylim[0], ylim[1], 100))
    z = draw_grid(problem.alpha, x, y)
    ui.axes.contourf(x, y, z, levels=[-float("inf"), 0], colors=["blue"],
                alpha=0.2)
    ui.axes.contourf(x, y, z, levels=[0, float("inf")], colors=["red"],
                alpha=0.2)

    if problem.inner_points:
        inner_points_array = array(problem.inner_points)
        ui.axes.scatter(inner_points_array[:, 0], inner_points_array[:, 1], c="blue", label="Inside")
    if problem.outer_points:
        outer_points_array = array(problem.outer_points)
        ui.axes.scatter(outer_points_array[:, 0], outer_points_array[:, 1], c="red", label="Outside")

    matrix = problem.alpha[:4].reshape(2, 2)
    eigenvalues = linalg.eigvals(matrix)
    ui.axes.set_title(f"Eigenvalues: {eigenvalues[0]:.2f}, {eigenvalues[1]:.2f}")
    if problem.inner_points or problem.outer_points:
        ui.axes.legend()
    ui.figure.canvas.draw_idle()


def on_click(event: MouseEvent, problem: Problem, ui: UI):
    """Handle mouse clicks."""
    if event.inaxes != ui.axes:
        return

    if ui.figure.canvas.toolbar.mode != "":
        return

    x = array([event.xdata, event.ydata])
    if event.button == 1:  # Left click
        problem.inner_points.append(x)
    elif event.button == 3:  # Right click
        problem.outer_points.append(x)

    problem.is_trained = False
    ui.button_train.set_active(True)
    draw_quadric(None, problem, ui)


def on_train(event: MouseEvent, problem: Problem, ui: UI):
    """Handle train button click."""
    steps = int(ui.text_steps.text)

    for _ in range(steps):
        if not train_step(problem):
            problem.is_trained = True
            ui.button_train.set_active(False)
            break

    draw_quadric(None, problem, ui)


def main():
    figure, axes = subplots(figsize=(5, 5))

    axes.set_xlim(-2, 2)
    axes.set_ylim(-2, 2)
    axes.grid(True)

    ax_train = figure.add_axes((0.5, 0.05, 0.1, 0.04))
    ax_steps = figure.add_axes((0.81, 0.05, 0.1, 0.04))

    ui = UI(
        figure=figure,
        axes=axes,
        button_train=Button(ax_train, "Train"),
        text_steps=TextBox(ax_steps, "Steps:", initial="1"),
    )
    problem = Problem()

    figure.canvas.mpl_connect("button_press_event", lambda event: on_click(event, problem, ui))
    figure.canvas.mpl_connect("draw_event", lambda event: draw_quadric(event, problem, ui))
    ui.button_train.on_clicked(lambda event: on_train(event, problem, ui))
    draw_quadric(None, problem, ui)

    show()


main()
