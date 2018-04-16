from local_linear_explanation import explanation_grid
from multilayer_perceptron import MultilayerPerceptron

from toy_colors import generate_dataset
from toy_colors import ignore_rule1
from toy_colors import ignore_rule2
from toy_colors import imgshape
from toy_colors import rule1_score
from toy_colors import rule2_score

import matplotlib.pyplot as plt


def explain(
    Xt,
    model,
    title='',
    length=4
):
    plt.title(title)
    explanation_grid(
        model.grad_explain(Xt[:length * length]),
        imgshape,
        length
    )
    plt.show()


def get_callback(clf, period=10):
    def callback(x, i, g):
        if i % period == 0:
            largest_gradient_mask = clf.largest_gradient_mask(x)
    return callback


def generate_regular_hypotheses(clf, inputs, n_hypotheses):
    hypotheses = [clf.largest_gradient_mask(inputs)]
    for i in range(n_hypotheses):
        hypotheses.append(
            clf.largest_gradient_mask(inputs) + sum(hypotheses)
        )
    return hypotheses


def main():
    X, Xt, y, yt = generate_dataset(cachefile='../data/toy-colors.npz')

    mlp = MultilayerPerceptron(l2_grads=1000)

    n_hypotheses = 3
    regular_hypotheses = generate_regular_hypotheses(
        clf=mlp,
        inputs=X,
        n_hypotheses=n_hypotheses
    )
    hypotheses = {
        'regular': regular_hypotheses,
    }

    mlp.fit(
        inputs=X,
        targets=y,
        hypotheses=hypotheses,
        callback=get_callback(mlp)
    )
    print(mlp.score(Xt, yt))
