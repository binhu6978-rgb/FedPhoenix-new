from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

from fedrad.config import FedRADConfig
from main_fedrad import parse_args


class FedPhoenixResetFidelityTests(unittest.TestCase):
    def test_verified_default_is_two_over_sixty_four(self):
        self.assertEqual(FedRADConfig().reset_ratio, 2.0 / 64.0)

    def test_cli_override_is_explicit_and_config_serializable(self):
        with patch.object(sys, "argv", ["main_fedrad.py", "--reset-ratio", "0.125"]):
            cli = parse_args()
        self.assertEqual(cli.reset_ratio, 0.125)
        self.assertEqual(FedRADConfig(reset_ratio=cli.reset_ratio).as_serializable_dict()["reset_ratio"], 0.125)


if __name__ == "__main__":
    unittest.main()
