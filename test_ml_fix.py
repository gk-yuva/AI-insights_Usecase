"""
Test to verify ML recommendations now have different probabilities
"""
import numpy as np
from ml_optimizer_wrapper import MLPortfolioOptimizer

def test_different_probabilities():
    """Verify that different feature vectors produce different probabilities"""
    
    try:
        optimizer = MLPortfolioOptimizer('trained_xgboost_model.json')
    except FileNotFoundError:
        print("ERROR: Model file not found. Please ensure trained_xgboost_model.json exists.")
        return False
    
    # Create test feature vectors with different values
    feature_vector_1 = np.array([
        # Investor features
        0.5, 0.5, 0.3, 0.7, 0.6, 20,
        # Asset features
        0.05, 0.02, 1.5, 1.2, 0.8, -0.1, 0.5, 2.0, 1.2,
        # Market features
        18.5, 0.15, 50, 22500, 0.02, 0.05, 0.8, 0.1, 0.055, 0.05, 0.02, 0.08,
        # Portfolio features
        20, 1000000, 0.3, 0.85, 0.05, 0.1, 0.02
    ])
    
    feature_vector_2 = np.array([
        # Investor features
        0.8, 0.7, 0.1, 0.9, 0.8, 30,
        # Asset features
        0.02, 0.015, 1.0, 0.9, 0.5, -0.15, 0.2, 1.5, 1.5,
        # Market features
        25.0, 0.20, 60, 23000, -0.02, 0.10, 0.6, 0.2, 0.065, 0.08, 0.03, 0.12,
        # Portfolio features
        30, 2000000, 0.5, 0.9, 0.08, 0.02, 0.00
    ])
    
    print("Testing ML model predictions with different feature vectors...\n")
    print("=" * 60)
    
    pred1 = optimizer.predict_recommendation_success(feature_vector_1)
    print("Prediction 1:")
    print(f"  Probability: {pred1['success_probability']:.4f}")
    print(f"  Score: {pred1['score']:.2f}/100")
    print(f"  Recommendation: {pred1['recommendation']}")
    print(f"  Model Decision: {pred1['model_decision']}")
    
    print("\n" + "=" * 60)
    
    pred2 = optimizer.predict_recommendation_success(feature_vector_2)
    print("Prediction 2:")
    print(f"  Probability: {pred2['success_probability']:.4f}")
    print(f"  Score: {pred2['score']:.2f}/100")
    print(f"  Recommendation: {pred2['recommendation']}")
    print(f"  Model Decision: {pred2['model_decision']}")
    
    print("\n" + "=" * 60)
    
    # Check if probabilities are different
    prob_difference = abs(pred1['success_probability'] - pred2['success_probability'])
    
    print(f"\nProbability Difference: {prob_difference:.4f}")
    print(f"Score Difference: {abs(pred1['score'] - pred2['score']):.2f}")
    
    if prob_difference > 0.01:
        print("\n✓ SUCCESS: Probabilities are now different!")
        print("✓ The ML recommendations fix is working correctly.")
        return True
    else:
        print("\n✗ FAILED: Probabilities are still the same or very close.")
        return False

if __name__ == "__main__":
    success = test_different_probabilities()
    exit(0 if success else 1)
