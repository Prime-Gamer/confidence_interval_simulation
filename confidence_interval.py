import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t
import pandas as pd

st.set_page_config(page_title="Confidence Interval Simulation", layout="wide")

st.title("Confidence Interval Simulation")

st.sidebar.header("Simulation Parameters")

# Input parameters
population_mean = st.sidebar.number_input("Population Mean", value=50.0)
population_std = st.sidebar.number_input("Population Standard Deviation", min_value=0.1, value=15.0)
sample_size = st.sidebar.number_input("Sample Size", min_value=2, value=50)
confidence_level = st.sidebar.slider("Confidence Level", min_value=0.8, max_value=0.99, value=0.95, step=0.01)
num_simulations = st.sidebar.slider("Number of Simulations", min_value=10, max_value=1000, value=100, step=10)

# Choose between using known sigma (z) or unknown sigma (t-distribution)
sigma_method = st.sidebar.radio(
    "Method",
    ["Known σ (z-distribution)", "Unknown σ (t-distribution)"]
)

# Run simulation
if st.sidebar.button("Run Simulation") or 'intervals' not in st.session_state:
    with st.spinner("Running simulations..."):
        # Z-score for the given confidence level
        if sigma_method == "Known σ (z-distribution)":
            critical_value = norm.ppf(1 - (1 - confidence_level) / 2)
            distribution_name = "z"
        else:
            critical_value = t.ppf(1 - (1 - confidence_level) / 2, df=sample_size-1)
            distribution_name = "t"
        
        # Store results
        intervals = []
        captured = 0
        
        for _ in range(num_simulations):
            # Generate sample
            sample = np.random.normal(loc=population_mean, scale=population_std, size=sample_size)
            sample_mean = np.mean(sample)
            
            if sigma_method == "Known σ (z-distribution)":
                # Use known population standard deviation
                margin_error = critical_value * (population_std / np.sqrt(sample_size))
            else:
                # Use sample standard deviation (t-distribution)
                sample_std = np.std(sample, ddof=1)  # ddof=1 for unbiased estimate
                margin_error = critical_value * (sample_std / np.sqrt(sample_size))
            
            lower_bound = sample_mean - margin_error
            upper_bound = sample_mean + margin_error
            intervals.append((lower_bound, sample_mean, upper_bound, margin_error))
            
            if lower_bound <= population_mean <= upper_bound:
                captured += 1
        
        st.session_state.intervals = intervals
        st.session_state.captured = captured
        st.session_state.critical_value = critical_value
        st.session_state.distribution_name = distribution_name

# Display results if available
if 'intervals' in st.session_state:
    intervals = st.session_state.intervals
    captured = st.session_state.captured
    critical_value = st.session_state.critical_value
    distribution_name = st.session_state.distribution_name
    
    col1, col2 = st.columns([7, 3])
    
    with col1:
        # Plotting
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, (low, mean, high, _) in enumerate(intervals):
            color = 'blue' if low <= population_mean <= high else 'red'
            alpha = 0.7 if color == 'blue' else 0.6
            ax.plot([i, i], [low, high], color=color, alpha=alpha)
            ax.plot(i, mean, 'ko', markersize=3)
        
        # Reference line
        ax.axhline(population_mean, color='green', linestyle='-', linewidth=2, label='Population Mean')
        
        ax.set_xlabel("Simulation Index")
        ax.set_ylabel("Confidence Interval")
        ax.set_title(f"Confidence Interval Simulation ({distribution_name}-distribution)\nCaptured Mean in {captured} of {num_simulations} Simulations ({captured/num_simulations*100:.1f}%)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
    
    with col2:
        st.subheader("Summary Statistics")
        
        # Calculate metrics
        coverage_rate = captured / num_simulations * 100
        
        # Display metrics
        st.metric("Coverage Rate", f"{coverage_rate:.1f}%", f"{coverage_rate - confidence_level*100:.1f}%")
        
        # Extract margins of error
        margins = [margin for _, _, _, margin in intervals]
        avg_margin = np.mean(margins)
        
        # Display additional metrics
        st.metric(f"Critical {distribution_name}-value", f"{critical_value:.4f}")
        st.metric("Avg. Margin of Error", f"{avg_margin:.4f}")
        
        # Calculate expected coverage
        expected_coverage = confidence_level * 100
        st.metric("Expected Coverage", f"{expected_coverage:.1f}%")
    
    # Display interval data
    st.subheader("Sample of Confidence Intervals")
    
    # Convert to DataFrame for better display
    intervals_df = pd.DataFrame(intervals, columns=["Lower", "Sample Mean", "Upper", "Margin of Error"])
    intervals_df["Contains True Mean"] = intervals_df.apply(
        lambda row: "Yes" if row["Lower"] <= population_mean <= row["Upper"] else "No", axis=1
    )
    
    # Show just first 10 rows
    st.dataframe(intervals_df.head(10))
    
    # Mathematical explanation
    st.subheader("Mathematical Explanation")
    
    formula_explanation = f"""
    **Formula Used:**
    
    {'Z-score' if distribution_name == 'z' else 'T-score'} calculation:
    - Confidence Level: {confidence_level:.2f}
    - Critical {distribution_name}-value: {critical_value:.4f}
    
    **Confidence Interval Formula:**
    For {distribution_name}-distribution:
    - CI = x̄ ± {distribution_name}(α/2) × {'σ' if distribution_name == 'z' else 's'}/√n
    - Where:
      - x̄ = sample mean
      - {'σ = population standard deviation' if distribution_name == 'z' else 's = sample standard deviation'}
      - n = sample size ({sample_size})
      - {distribution_name}(α/2) = critical value ({critical_value:.4f})
    """
    
    st.markdown(formula_explanation)