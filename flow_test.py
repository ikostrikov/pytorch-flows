import unittest

import torch

import flows as fnn

EPS = 1e-5
BATCH_SIZE = 32
NUM_INPUTS = 11
NUM_HIDDEN = 64


class TestFlow(unittest.TestCase):
    def testCoupling(self):
        m1 = fnn.FlowSequential(fnn.CouplingLayer(NUM_INPUTS, NUM_HIDDEN))

        x = torch.randn(BATCH_SIZE, NUM_INPUTS)

        y, logdets = m1(x)
        z, inv_logdets = m1(y, mode='inverse')

        self.assertTrue((logdets + inv_logdets).abs().max() < EPS,
                        'CouplingLayer Det is not zero.')
        self.assertTrue((x - z).abs().max() < EPS, 'CouplingLayer is wrong')

    def testInv(self):
        m1 = fnn.FlowSequential(fnn.InvertibleMM(NUM_INPUTS))

        x = torch.randn(BATCH_SIZE, NUM_INPUTS)

        y, logdets = m1(x)
        z, inv_logdets = m1(y, mode='inverse')

        self.assertTrue((logdets + inv_logdets).abs().max() < EPS, 'InvMM Det is not zero.')
        self.assertTrue((x - z).abs().max() < EPS, 'InvMM is wrong.')

    def testActNorm(self):
        m1 = fnn.FlowSequential(fnn.ActNorm(NUM_INPUTS))

        x = torch.randn(BATCH_SIZE, NUM_INPUTS)

        y, logdets = m1(x)
        z, inv_logdets = m1(y, mode='inverse')

        self.assertTrue((logdets + inv_logdets).abs().max() < EPS, 'ActNorm Det is not zero.')
        self.assertTrue((x - z).abs().max() < EPS, 'ActNorm is wrong.')

        # Second run.
        x = torch.randn(BATCH_SIZE, NUM_INPUTS)

        y, logdets = m1(x)
        z, inv_logdets = m1(y, mode='inverse')

        self.assertTrue((logdets + inv_logdets).abs().max() < EPS,
                        'ActNorm Det is not zero for the second run.')
        self.assertTrue((x - z).abs().max() < EPS,
                        'ActNorm is wrong for the second run.')

    def testBatchNorm(self):
        m1 = fnn.FlowSequential(fnn.BatchNormFlow(NUM_INPUTS))
        m1.train()

        x = torch.randn(BATCH_SIZE, NUM_INPUTS)

        y, logdets = m1(x)
        z, inv_logdets = m1(y, mode='inverse')

        self.assertTrue((logdets + inv_logdets).abs().max() < EPS,
                        'BatchNorm Det is not zero.')
        self.assertTrue((x - z).abs().max() < EPS, 'BatchNorm is wrong.')

        # Second run.
        x = torch.randn(BATCH_SIZE, NUM_INPUTS)

        y, logdets = m1(x)
        z, inv_logdets = m1(y, mode='inverse')

        self.assertTrue((logdets + inv_logdets).abs().max() < EPS,
                        'BatchNorm Det is not zero for the second run.')
        self.assertTrue((x - z).abs().max() < EPS,
                        'BatchNorm is wrong for the second run.')

        m1.eval()
        m1 = fnn.FlowSequential(fnn.BatchNormFlow(NUM_INPUTS))

        x = torch.randn(BATCH_SIZE, NUM_INPUTS)

        y, logdets = m1(x)
        z, inv_logdets = m1(y, mode='inverse')

        self.assertTrue((logdets + inv_logdets).abs().max() < EPS,
                        'BatchNorm Det is not zero in eval.')
        self.assertTrue((x - z).abs().max() < EPS,
                        'BatchNorm is wrong in eval.')

    def testSequential(self):
        m1 = fnn.FlowSequential(
            fnn.ActNorm(NUM_INPUTS), fnn.InvertibleMM(NUM_INPUTS),
            fnn.CouplingLayer(NUM_INPUTS, NUM_HIDDEN))

        x = torch.randn(BATCH_SIZE, NUM_INPUTS)

        y, logdets = m1(x)
        z, inv_logdets = m1(y, mode='inverse')

        self.assertTrue((logdets + inv_logdets).abs().max() < EPS, 'ActNorm Det is not zero.')
        self.assertTrue((x - z).abs().max() < EPS, 'ActNorm is wrong.')

        # Second run.
        x = torch.randn(BATCH_SIZE, NUM_INPUTS)

        y, logdets = m1(x)
        z, inv_logdets = m1(y, mode='inverse')

        self.assertTrue((logdets + inv_logdets).abs().max() < EPS,
                        'ActNorm Det is not zero for the second run.')
        self.assertTrue((x - z).abs().max() < EPS,
                        'ActNorm is wrong for the second run.')

    def testSequentialBN(self):
        m1 = fnn.FlowSequential(
            fnn.BatchNormFlow(NUM_INPUTS), fnn.InvertibleMM(NUM_INPUTS),
            fnn.CouplingLayer(NUM_INPUTS, NUM_HIDDEN))

        m1.train()
        x = torch.randn(BATCH_SIZE, NUM_INPUTS)

        y, logdets = m1(x)
        z, inv_logdets = m1(y, mode='inverse')

        self.assertTrue((logdets + inv_logdets).abs().max() < EPS,
                        'Sequential Det is not zero.')
        self.assertTrue((x - z).abs().max() < EPS, 'Sequential is wrong.')

        # Second run.
        x = torch.randn(BATCH_SIZE, NUM_INPUTS)

        y, logdets = m1(x)
        z, inv_logdets = m1(y, mode='inverse')

        self.assertTrue((logdets + inv_logdets).abs().max() < EPS,
                        'Sequential Det is not zero for the second run.')
        self.assertTrue((x - z).abs().max() < EPS,
                        'Sequential is wrong for the second run.')

        m1.eval()
        # Eval run.
        x = torch.randn(BATCH_SIZE, NUM_INPUTS)

        y, logdets = m1(x)
        z, inv_logdets = m1(y, mode='inverse')

        self.assertTrue((logdets + inv_logdets).abs().max() < EPS,
                        'Sequential Det is not zero for the eval run.')
        self.assertTrue((x - z).abs().max() < EPS,
                        'Sequential is wrong for the eval run.')


if __name__ == "__main__":
    unittest.main()
