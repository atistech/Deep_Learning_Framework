import unittest
import activations as a

class TestActivationFunctions(unittest.TestCase):

    def test_linear(self):
        self.assertEqual(a.linear(10), 10)

    def test_logsigmoid(self):
        self.assertEqual(a.logsigmoid(0), 0.5)

if __name__ == '__main__':
    unittest.main()