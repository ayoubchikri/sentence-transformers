from __future__ import annotations

import pytest
import torch
from torch import nn

from sentence_transformers.sentence_transformer.losses import BatchAllTripletLoss
from sentence_transformers.sentence_transformer.losses.batch_hard_triplet import BatchHardTripletLossDistanceFunction


def _reference_loss(embeddings, labels, distance_metric, margin):
    distances = distance_metric(embeddings)
    losses = []
    for anchor in range(len(labels)):
        for positive in range(len(labels)):
            if anchor == positive or labels[anchor] != labels[positive]:
                continue
            for negative in range(len(labels)):
                if labels[anchor] != labels[negative]:
                    value = distances[anchor, positive] - distances[anchor, negative] + margin
                    # The existing loss retains the hinge's gradient at exactly zero.
                    losses.append(torch.where(value < 0, 0.0, value))
    if not losses:
        return embeddings.sum() * 0
    losses = torch.stack(losses)
    return losses.sum() / ((losses > 1e-16).sum() + 1e-16)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "distance_metric",
    [BatchHardTripletLossDistanceFunction.euclidean_distance, BatchHardTripletLossDistanceFunction.cosine_distance],
    ids=["euclidean", "cosine"],
)
@pytest.mark.parametrize("case", ["balanced", "uneven", "no_positives", "no_negatives", "ties"])
def test_batch_all_matches_explicit_triplets(dtype, distance_metric, case):
    generator = torch.Generator().manual_seed(42)
    embeddings = torch.randn(8, 4, generator=generator, dtype=dtype)
    labels = torch.arange(8) // 2
    if case == "uneven":
        labels = torch.tensor([0, 0, 0, 1, 1, 2, 3, 3])
    elif case == "no_positives":
        labels = torch.arange(8)
    elif case == "no_negatives":
        labels = torch.zeros(8, dtype=torch.long)
    elif case == "ties":
        embeddings = torch.tensor([[1, 0], [1, 0], [-1, 0], [-1, 0], [0, 1], [0, 1], [0, -1], [0, -1]], dtype=dtype)
    embeddings.requires_grad_()
    reference_embeddings = embeddings.detach().clone().requires_grad_()
    loss_fn = BatchAllTripletLoss(nn.Identity(), distance_metric=distance_metric, margin=1.0)

    actual = loss_fn.compute_loss_from_embeddings([embeddings], labels)
    expected = _reference_loss(reference_embeddings, labels, distance_metric, margin=1.0)
    actual.backward()
    expected.backward()

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(embeddings.grad, reference_embeddings.grad)
    assert torch.isfinite(actual)
    assert torch.isfinite(embeddings.grad).all()


def test_batch_all_does_not_save_a_cube_for_paired_labels():
    batch_size = 32
    embeddings = torch.randn(batch_size, 4, requires_grad=True)
    labels = torch.arange(batch_size) // 2
    saved_sizes = []

    def pack(tensor):
        saved_sizes.append(tensor.numel())
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        loss = BatchAllTripletLoss(nn.Identity()).compute_loss_from_embeddings([embeddings], labels)
    loss.backward()

    assert saved_sizes
    assert max(saved_sizes) <= batch_size**2
    assert embeddings.grad.abs().sum() > 0


def test_batch_all_encoder_updates_match_explicit_triplets():
    torch.manual_seed(7)
    encoder = nn.Linear(4, 3)
    reference_encoder = nn.Linear(4, 3)
    reference_encoder.load_state_dict(encoder.state_dict())
    optimizer = torch.optim.SGD(encoder.parameters(), lr=0.01)
    reference_optimizer = torch.optim.SGD(reference_encoder.parameters(), lr=0.01)
    inputs = torch.randn(8, 4)
    labels = torch.arange(8) // 2
    loss_fn = BatchAllTripletLoss(nn.Identity(), margin=1.0)

    for _ in range(3):
        optimizer.zero_grad()
        reference_optimizer.zero_grad()
        loss_fn.compute_loss_from_embeddings([encoder(inputs)], labels).backward()
        _reference_loss(reference_encoder(inputs), labels, loss_fn.distance_metric, margin=1.0).backward()
        optimizer.step()
        reference_optimizer.step()
        for actual, expected in zip(encoder.parameters(), reference_encoder.parameters()):
            torch.testing.assert_close(actual, expected)
