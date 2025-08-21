"""
Unit tests for risk management functionality
"""

import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.risk_management import (
    validate_percentage, RiskParameterError,
    calculate_position_size_by_risk
)

class TestRiskManagement(unittest.TestCase):
    """Test risk management functions"""
    
    def test_validate_percentage_valid(self):
        """Test valid percentage validation"""
        result = validate_percentage(0.05, "test_param", 0.01, 0.1)
        self.assertEqual(result, 0.05)
    
    def test_validate_percentage_too_low(self):
        """Test percentage below minimum"""
        with self.assertRaises(RiskParameterError):
            validate_percentage(0.001, "test_param", 0.01, 0.1)
    
    def test_validate_percentage_too_high(self):
        """Test percentage above maximum"""
        with self.assertRaises(RiskParameterError):
            validate_percentage(0.2, "test_param", 0.01, 0.1)
    
    def test_validate_percentage_not_number(self):
        """Test non-numeric percentage"""
        with self.assertRaises(RiskParameterError):
            validate_percentage("5%", "test_param", 0.01, 0.1)
    
    def test_position_size_calculation(self):
        """Test position size calculation based on risk"""
        portfolio_value = 10000
        entry_price = 100
        stop_loss_price = 95
        max_risk = 0.02  # 2% risk
        
        expected_size = (portfolio_value * max_risk) / (entry_price - stop_loss_price)
        actual_size = calculate_position_size_by_risk(
            portfolio_value, entry_price, stop_loss_price, max_risk
        )
        
        self.assertEqual(actual_size, expected_size)
    
    def test_position_size_zero_risk(self):
        """Test position size with zero price risk"""
        portfolio_value = 10000
        entry_price = 100
        stop_loss_price = 100  # Same as entry
        
        size = calculate_position_size_by_risk(
            portfolio_value, entry_price, stop_loss_price
        )
        
        self.assertEqual(size, 0)

if __name__ == '__main__':
    unittest.main()