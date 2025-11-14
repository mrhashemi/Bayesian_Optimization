"""
BAYESIAN OPTIMIZATION FROM SCRATCH
===================================

A minimal, self-contained implementation demonstrating core concepts:

1. Gaussian Process (GP) as a surrogate model
2. Different acquisition functions: 
    Expected Improvement (EI), 
    Probability of Improvement (PI), 
    Upper Confidence Bound (UCB), 
    Log Expected Improvement (Log-EI)
3. Sequential optimization loop

Key Mathematical Components:
----------------------------
• GP Prior: f ~ GP(m(x), k(x,x'))
• Kernel: k(x,x') = σ² exp(-||x-x'||²/(2ℓ²))  [RBF/Squared Exponential]
• Posterior Mean: μ(x|D) = K(x,X) [K(X,X) + σ_n²I]^(-1) y
• Posterior Variance: σ²(x|D) = k(x,x) - K(x,X) [K(X,X) + σ_n²I]^(-1) K(X,x)
• Expected Improvement: EI(x) = (μ - f_best - ξ)Φ(Z) + σφ(Z)

Dependencies: numpy, scipy
Based on https://arxiv.org/abs/1807.02811
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# =============================================================================
# OPTIMIZATION ALGORITHMS
# =============================================================================

class BFGSOptim():
    """
    BFGS Optimization Algorithm.
    
    Based on: https://github.com/mrhashemi/Optimizers/blob/main/adam_bfgs_optimization.ipynb
    """
    def __init__(self, w0, grad_error=1.0e-6):
        """Initialize BFGS optimizer"""
        self.a = 1.0
        self.c1 = 1.0e-4 
        self.c2 = 0.9

        self.d = w0.size 
        self.H = np.eye(self.d)
        self.w = w0.copy()
        self.w_new = w0.copy()
        self.it = 2

        self.error_tolerance = grad_error

    def line_search(self, func, grad_func, max_line_search_iter=100):
        """Backtracking line search with Wolfe conditions"""
        self.a = 1.0
        self.grad = grad_func(self.w)
        self.p = -self.H @ self.grad  # search direction (Newton Method)
        f = func(self.w)
        self.w_new = self.w + self.a * self.p
        self.grad_new = grad_func(self.w_new)

        # Wolfe conditions with iteration limit
        line_search_iter = 0
        while func(self.w_new) >= f + (self.c1 * self.a * self.grad.T @ self.p) or \
              self.grad_new.T @ self.p <= self.c2 * self.grad.T @ self.p: 
            self.a *= 0.5
            self.w_new = self.w + self.a * self.p
            self.grad_new = grad_func(self.w_new)
            
            line_search_iter += 1
            if line_search_iter >= max_line_search_iter:
                # Line search failed to converge, return current step
                break

        return self.a

    def get_updated_weights(self, func, grad_func, w0):
        """Update weights using BFGS update rule"""
        self.w = w0.copy()
        self.it += 1
        self.a = self.line_search(func, grad_func)
        
        s = self.a * self.p
        s = np.reshape(s, (self.d, 1))
        
        y = self.grad_new - self.grad
        y = np.reshape(y, (self.d, 1))
        
        # Check for division by zero (curvature condition)
        y_dot_s = y.T @ s
        
        # Skip BFGS update if curvature is too small
        if abs(y_dot_s) < 1e-10:
            # Don't update Hessian approximation, just return new point
            return self.w_new
        
        r = 1 / y_dot_s
        li = np.eye(self.d) - r * (s @ y.T)
        ri = np.eye(self.d) - r * (y @ s.T)

        hess_inter = li @ self.H @ ri
        self.H = hess_inter + r * (s @ s.T)  # BFGS Update
        self.grad = self.grad_new.copy()
    
        return self.w_new

    def check_convergence(self):
        """Check if gradient norm is below tolerance"""
        return np.linalg.norm(self.grad) <= self.error_tolerance


def compute_gradient(func, x, h=None):
    """
    Central finite difference gradient calculation.
    
    Parameters
    ----------
    func : callable
        Function to differentiate
    x : ndarray
        Point at which to evaluate gradient
    h : float, optional
        Step size (default: cubic root of machine epsilon)
    
    Returns
    -------
    nabla : ndarray
        Gradient vector
    """
    if h is None:
        h = np.cbrt(np.finfo(float).eps)
    
    nabla = np.zeros_like(x)
    d = x.size
    for i in range(d): 
        x_for = x.copy()
        x_back = x.copy()
        x_for[i] += h 
        x_back[i] -= h 
        nabla[i] = (func(x_for) - func(x_back)) / (2 * h) 
    return nabla


# =============================================================================
# GAUSSIAN PROCESS IMPLEMENTATION
# =============================================================================

class SimpleGaussianProcess:
    """
    Minimal Gaussian Process Regression implementation.

    Uses Radial Basis Function (RBF) kernel for smoothness.
    Provides closed-form posterior mean and variance.

    Parameters
    ----------
    length_scale : float
        Kernel length-scale parameter ℓ (controls smoothness)
    signal_variance : float
        Kernel signal variance σ² (controls output scale)
    noise_variance : float
        Observation noise variance σ_n²
    """

    def __init__(self, length_scale=1.0, signal_variance=1.0, noise_variance=1e-6):
        self.length_scale = length_scale
        self.signal_variance = signal_variance
        self.noise_variance = noise_variance

        # Stored after fitting
        self.X_train = None
        self.y_train = None
        self.K_inv = None  # Inverse of augmented kernel matrix

    def rbf_kernel(self, X1, X2):
        """
        Radial Basis Function (Squared Exponential) Kernel.

        k(x, x') = σ² * exp(-||x - x'||² / (2ℓ²))

        Parameters
        ----------
        X1, X2 : array-like, shape (n_samples,)
            Input points

        Returns
        -------
        K : ndarray, shape (n_samples_1, n_samples_2)
            Kernel matrix
        """
        X1 = X1.reshape(-1, 1)
        X2 = X2.reshape(-1, 1)

        # Compute squared Euclidean distances efficiently
        sqdist = (np.sum(X1**2, 1).reshape(-1, 1) + 
                  np.sum(X2**2, 1) - 
                  2 * np.dot(X1, X2.T))

        return self.signal_variance * np.exp(-0.5 / self.length_scale**2 * sqdist)

    def fit(self, X_train, y_train):
        """
        Fit GP to training data by computing posterior distribution.

        Stores the inverse of K_y = K(X,X) + σ_n²I for efficient prediction.

        Parameters
        ----------
        X_train : array-like, shape (n_samples,)
            Training input points
        y_train : array-like, shape (n_samples,)
            Training output values (observations)
        """
        self.X_train = X_train
        self.y_train = y_train

        # Compute kernel matrix K(X, X)
        K = self.rbf_kernel(X_train, X_train)

        # Add observation noise: K_y = K + σ_n²I
        K_y = K + self.noise_variance * np.eye(len(X_train))

        # Compute and store inverse for predictions
        # Note: In production, use Cholesky decomposition for stability
        self.K_inv = np.linalg.inv(K_y)

    def predict(self, X_test):
        """
        Compute posterior predictive distribution at test points.

        Returns Gaussian distribution: N(μ(x*|D), σ²(x*|D))

        Parameters
        ----------
        X_test : array-like, shape (n_test,)
            Test input points

        Returns
        -------
        mean : ndarray, shape (n_test,)
            Posterior mean μ(x*|D)
        std : ndarray, shape (n_test,)
            Posterior standard deviation σ(x*|D)
        """
        # Cross-covariance between test and training points
        K_star = self.rbf_kernel(X_test, self.X_train)

        # Auto-covariance at test points
        K_star_star = self.rbf_kernel(X_test, X_test)

        # Posterior mean: μ = K* K_y^(-1) y
        mean = K_star @ self.K_inv @ self.y_train

        # Posterior variance: σ² = K** - K* K_y^(-1) K*^T
        variance = np.diag(K_star_star - K_star @ self.K_inv @ K_star.T)

        # Ensure numerical stability (variance must be positive)
        variance = np.maximum(variance, 1e-9)
        std = np.sqrt(variance)

        return mean, std


# =============================================================================
# ACQUISITION FUNCTION CLASS
# =============================================================================

class AcquisitionFunctions:
    @staticmethod
    def expected_improvement(mu, std, f_best, xi=0.01):
        std = np.maximum(std, 1e-12)
        improvement = mu - f_best - xi
        Z = improvement / std
        ei = improvement * norm.cdf(Z) + std * norm.pdf(Z)
        return ei

    @staticmethod
    def probability_of_improvement(mu, std, f_best, xi=0.01):
        std = np.maximum(std, 1e-12)
        improvement = mu - f_best - xi
        Z = improvement / std
        return norm.cdf(Z)

    @staticmethod
    def upper_confidence_bound(mu, std, kappa=2.0):
        return mu + kappa * std

    @staticmethod
    def log_expected_improvement(mu, std, f_best, xi=0.01):
        std = np.maximum(std, 1e-12)
        improvement = mu - f_best - xi
        Z = improvement / std
        log_ei = np.where(
            Z > -10,
            np.log(std) + np.log(norm.cdf(Z)) + \
            np.log1p(Z * np.exp(norm.logpdf(Z) - np.log(norm.cdf(Z)))),
            norm.logpdf(Z) + np.log(std)
        )
        return log_ei
    # Note: log_expected_improvement is best used for logging or as a fallback when EI is extremely small,
    # not as the main acquisition function for optimization. It can be numerically unstable and produce a 
    # flat or erratic surface.


# =============================================================================
# VISUALIZATION FUNCTION
# =============================================================================

def plot_bo_iteration(iteration, X_observed, y_observed, gp, bounds, objective_func,
                     x_next=None, best_x=None, best_y=None, xi=0.01):
    """
    Visualize Bayesian Optimization state at a given iteration.
    
    Creates a 2-panel plot showing:
    - Top: GP posterior (mean, uncertainty) and observations
    - Bottom: Acquisition function (Expected Improvement)
    
    Parameters
    ----------
    iteration : int
        Current iteration number
    X_observed : ndarray
        All observed points so far
    y_observed : ndarray
        All observed values so far
    gp : SimpleGaussianProcess
        Fitted GP model
    bounds : tuple
        Search space (x_min, x_max)
    objective_func : callable
        True objective function (for visualization only)
    x_next : float, optional
        Next point to sample (from acquisition optimization)
    best_x : float, optional
        Best point found so far
    best_y : float, optional
        Best value found so far
    xi : float
        Exploration parameter
    """
    x_min, x_max = bounds
    
    # Dense grid for plotting
    X_plot = np.linspace(x_min, x_max, 500)
    
    # Get GP predictions
    mean, std = gp.predict(X_plot)
    
    # Compute true function values (for reference)
    y_true = np.array([objective_func(x) for x in X_plot])
    
    # Compute acquisition function
    mean_plot, std_plot = gp.predict(X_plot)
    acq_plot = AcquisitionFunctions.expected_improvement(mean_plot, std_plot, best_y, xi=xi)
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # =========================================================================
    # TOP PANEL: Gaussian Process Posterior
    # =========================================================================
    
    # Plot true function (dashed line for reference)
    ax1.plot(X_plot, y_true, 'k--', alpha=0.3, label='True Function', linewidth=1.5)
    
    # Plot GP mean
    ax1.plot(X_plot, mean, 'b-', label='GP Mean μ(x)', linewidth=2)
    
    # Plot uncertainty (±2σ confidence band)
    ax1.fill_between(X_plot, mean - 2*std, mean + 2*std, 
                     alpha=0.2, color='blue', label='±2σ Uncertainty')
    
    # Plot observed points
    ax1.scatter(X_observed, y_observed, c='red', s=100, zorder=5, 
               marker='o', edgecolors='darkred', linewidth=1.5,
               label=f'Observations ({len(X_observed)})')
    
    # Highlight best point found so far
    if best_x is not None and best_y is not None:
        ax1.scatter([best_x], [best_y], c='gold', s=300, zorder=6,
                   marker='*', edgecolors='orange', linewidth=2,
                   label=f'Best (f={best_y:.3f})')
    
    # Mark next sampling point
    if x_next is not None:
        y_next_pred, _ = gp.predict(np.array([x_next]))
        ax1.axvline(x_next, color='green', linestyle=':', linewidth=2, alpha=0.7)
        ax1.scatter([x_next], [y_next_pred], c='lime', s=200, zorder=7,
                   marker='^', edgecolors='darkgreen', linewidth=2,
                   label=f'Next Sample (x={x_next:.3f})')
    
    ax1.set_ylabel('f(x)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Bayesian Optimization - Iteration {iteration}', 
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # =========================================================================
    # BOTTOM PANEL: Acquisition FUNCTION (EXPECTED IMPROVEMENT)
    # =========================================================================
    
    # Plot EI
    ax2.plot(X_plot, acq_plot, 'g-', linewidth=2, label='Expected Improvement')
    ax2.fill_between(X_plot, 0, acq_plot, alpha=0.3, color='green')
    
    # Mark maximum of EI (next sampling point)
    if x_next is not None:
        mean_next, std_next = gp.predict(np.array([x_next]))
        ei_max = AcquisitionFunctions.expected_improvement(mean_next, std_next, best_y, xi=xi)[0]
        ax2.scatter([x_next], [ei_max], c='lime', s=200, zorder=5,
                   marker='^', edgecolors='darkgreen', linewidth=2,
                   label=f'Max EI at x={x_next:.3f}')
        ax2.axvline(x_next, color='green', linestyle=':', linewidth=2, alpha=0.7)
    
    # Mark current observations on x-axis
    ax2.scatter(X_observed, np.zeros_like(X_observed), c='red', s=50, 
               zorder=4, marker='o', alpha=0.5)
    
    ax2.set_xlabel('x', fontsize=12, fontweight='bold')
    ax2.set_ylabel('EI(x)', fontsize=12, fontweight='bold')
    ax2.set_title('Acquisition Function (Expected Improvement)', 
                 fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(x_min, x_max)
    
    plt.tight_layout()
    plt.pause(0.1)  # Small pause to update display
    
    return fig


# =============================================================================
# MAIN BAYESIAN OPTIMIZATION ALGORITHM
# =============================================================================

def bayesian_optimization(objective_func, bounds, n_initial=5, n_iterations=20, 
                         xi=0.01, verbose=True, plot=False, plot_interval=3,
                         convergence_threshold=1e-4, acq_type='expected_improvement'):
    """
    Bayesian Optimization main loop.

    Algorithm:
    ----------
    1. Initialize: Sample n_initial random points
    2. Loop for n_iterations:
        a. Fit GP to all observed data
        b. Optimize acquisition function to find next sample point
        c. Evaluate objective function at new point
        d. Update dataset with new observation
        e. Check convergence criteria
    3. Return best point found

    Parameters
    ----------
    objective_func : callable
        Black-box function to maximize: f: R^d -> R
    bounds : tuple
        Search space bounds (x_min, x_max)
    n_initial : int, default=5
        Number of random initialization samples
    n_iterations : int, default=20
        Number of Bayesian optimization iterations
    xi : float, default=0.01
        Exploration parameter for Expected Improvement
    verbose : bool, default=True
        Print progress information
    plot : bool, default=False
        If True, display visualization at each iteration
    plot_interval : int, default=3
        Create a new plot figure every N iterations
    convergence_threshold : float, default=1e-4
        Stop optimization if improvement in best value is below this threshold
    acq_type : str, default='expected_improvement'
        Acquisition function to use. Options: 'expected_improvement', 'probability_of_improvement', 'upper_confidence_bound', 'log_expected_improvement'

    Returns
    -------
    X_observed : ndarray
        All sampled points
    y_observed : ndarray
        All observed function values
    best_x : float
        Best point found
    best_y : float
        Best function value found
    """
    x_min, x_max = bounds

    # Storage for all observations
    X_observed = []
    y_observed = []

    if verbose:
        print("="*70)
        print("BAYESIAN OPTIMIZATION")
        print("="*70)

    # =========================================================================
    # STEP 1: INITIALIZATION - Random Sampling
    # =========================================================================
    if verbose:
        print(f"\n[INITIALIZATION] Sampling {n_initial} random points...")

    for i in range(n_initial):
        x = np.random.uniform(x_min, x_max)
        y = objective_func(x)
        X_observed.append(x)
        y_observed.append(y)

        if verbose:
            print(f"  Sample {i+1}: x={x:.4f}, f(x)={y:.4f}")

    X_observed = np.array(X_observed)
    y_observed = np.array(y_observed)

    # Track best observed value
    best_idx = np.argmax(y_observed)
    best_x = X_observed[best_idx]
    best_y = y_observed[best_idx]

    if verbose:
        print(f"\nInitial best: x={best_x:.4f}, f(x)={best_y:.4f}")

    # Setup plotting if requested
    if plot:
        all_figures = []  # Store all figure references

    # =========================================================================
    # STEP 2: OPTIMIZATION LOOP
    # =========================================================================
    if verbose:
        print(f"\n[OPTIMIZATION] Running up to {n_iterations} iterations...")
        print(f"Convergence threshold: {convergence_threshold}")
        print("-"*70)

    # Choose acquisition function here:
    acq_kwargs = {'xi': xi}  # For UCB, use {'kappa': 2.0}

    for iteration in range(n_iterations):
        if verbose:
            print(f"\nIteration {iteration + 1}/{n_iterations}")

        # ---------------------------------------------------------------------
        # 2a. Fit Gaussian Process to current data
        # ---------------------------------------------------------------------
        gp = SimpleGaussianProcess(length_scale=0.5, signal_variance=2.0)
        gp.fit(X_observed, y_observed)

        if verbose:
            print(f"  GP fitted to {len(X_observed)} observations")

        # ---------------------------------------------------------------------
        # 2b. Optimize Expected Improvement acquisition function
        # ---------------------------------------------------------------------
        # Strategy: Grid search for initialization + BFGS local refinement

        # Coarse grid search
        X_grid = np.linspace(x_min, x_max, 100)
        mean_grid, std_grid = gp.predict(X_grid)
        acq_values = AcquisitionFunctions.expected_improvement(mean_grid, std_grid, best_y, xi=xi)
        best_grid_idx = np.argmax(acq_values)
        x0 = X_grid[best_grid_idx]

        # Fine-tune with BFGS optimization (maximize EI = minimize negative EI)
        def neg_acq(x):
            # Use the selected acquisition function
            x_val = float(x) if np.size(x) == 1 else x[0]
            mean, std = gp.predict(np.array([x_val]))
            if acq_type == 'expected_improvement':
                val = AcquisitionFunctions.expected_improvement(mean, std, best_y, **acq_kwargs)[0]
            elif acq_type == 'probability_of_improvement':
                val = AcquisitionFunctions.probability_of_improvement(mean, std, best_y, **acq_kwargs)[0]
            elif acq_type == 'upper_confidence_bound':
                val = AcquisitionFunctions.upper_confidence_bound(mean, std, **acq_kwargs)[0]
            elif acq_type == 'log_expected_improvement':
                val = AcquisitionFunctions.log_expected_improvement(mean, std, best_y, **acq_kwargs)[0]
            else:
                raise ValueError(f"Unknown acquisition type: {acq_type}")
            return -val

        def grad_neg_acq(x):
            return compute_gradient(neg_acq, x)

        # Initialize BFGS optimizer with grid search result
        x_init = np.array([x0])
        bfgs_optimizer = BFGSOptim(x_init, grad_error=1e-5)
        
        # Run BFGS optimization
        x_opt = x_init.copy()
        converged = False
        max_bfgs_iter = 50
        
        if verbose:
            print(f"  Starting BFGS optimization from x0={x0:.4f}")
        
        for bfgs_iter in range(max_bfgs_iter):
            try:
                x_opt_prev = x_opt.copy()
                x_opt = bfgs_optimizer.get_updated_weights(neg_acq, grad_neg_acq, x_opt)
                converged = bfgs_optimizer.check_convergence()
                
                # Clip to bounds
                x_opt = np.clip(x_opt, x_min, x_max)
                
                if verbose and (bfgs_iter % 10 == 0 or bfgs_iter < 3):
                    print(f"    BFGS iter {bfgs_iter}: x={x_opt[0]:.6f}, step={abs(x_opt[0]-x_opt_prev[0]):.6e}, grad_norm={np.linalg.norm(bfgs_optimizer.grad):.6e}")
                
                if converged:
                    if verbose:
                        print(f"    BFGS converged at iteration {bfgs_iter}")
                    break
                    
                # Additional safety check: if step is too small, exit
                if abs(x_opt[0] - x_opt_prev[0]) < 1e-10:
                    if verbose:
                        print(f"    BFGS: step size too small, stopping at iteration {bfgs_iter}")
                    break
                    
            except Exception as e:
                if verbose:
                    print(f"    BFGS error at iteration {bfgs_iter}: {e}")
                break
        
        x_next = float(x_opt[0])

        if verbose:
            print(f"  Acquisition optimized: x_next={x_next:.4f}")

        # ---------------------------------------------------------------------
        # Visualization (before evaluation) - create new plot every N iterations
        # ---------------------------------------------------------------------
        if plot and ((iteration + 1) % plot_interval == 0 or iteration == n_iterations - 1):
            fig = plot_bo_iteration(
                iteration=iteration + 1,
                X_observed=X_observed,
                y_observed=y_observed,
                gp=gp,
                bounds=bounds,
                objective_func=objective_func,
                x_next=x_next,
                best_x=best_x,
                best_y=best_y,
                xi=xi
            )
            all_figures.append(fig)
            if verbose:
                print(f"  Plot created for iteration {iteration + 1}")

        # ---------------------------------------------------------------------
        # 2c. Evaluate objective function at selected point
        # ---------------------------------------------------------------------
        y_next = objective_func(x_next)

        if verbose:
            print(f"  Objective evaluated: f(x_next)={y_next:.4f}")

        # ---------------------------------------------------------------------
        # 2e. Check convergence criteria (before updating best)
        # ---------------------------------------------------------------------
        # Calculate change between newly sampled point and current best
        position_change = abs(x_next - best_x)
        
        if position_change < convergence_threshold:
            if verbose:
                print(f"\n{'='*70}")
                print(f"CONVERGENCE REACHED at iteration {iteration + 1}")
                print(f"Position change ({position_change:.6e}) < threshold ({convergence_threshold:.6e})")
                print(f"x_next={x_next:.6f}, best_x={best_x:.6f}")
                print(f"{'='*70}")
            break

        # ---------------------------------------------------------------------
        # 2d. Update dataset with new observation
        # ---------------------------------------------------------------------
        X_observed = np.append(X_observed, x_next)
        y_observed = np.append(y_observed, y_next)

        # Update best if improved
        if y_next > best_y:
            best_x = x_next
            best_y = y_next
            if verbose:
                print(f"  >> NEW BEST: x={best_x:.4f}, f(x)={best_y:.4f}")
        else:
            if verbose:
                print(f"  Current best: x={best_x:.4f}, f(x)={best_y:.4f}")
        
    # =========================================================================
    # STEP 3: RETURN RESULTS
    # =========================================================================
    if verbose:
        print("\n" + "="*70)
        print("OPTIMIZATION COMPLETE")
        print("="*70)
        print(f"\nBest point found: x* = {best_x:.6f}")
        print(f"Best value found: f(x*) = {best_y:.6f}")
        print(f"Total evaluations: {len(X_observed)}")

    # Final plot showing the complete optimization
    if plot:
        # Refit GP with all data for final visualization
        gp_final = SimpleGaussianProcess(length_scale=0.5, signal_variance=2.0)
        gp_final.fit(X_observed, y_observed)
        
        fig_final = plot_bo_iteration(
            iteration=n_iterations,
            X_observed=X_observed,
            y_observed=y_observed,
            gp=gp_final,
            bounds=bounds,
            objective_func=objective_func,
            x_next=None,
            best_x=best_x,
            best_y=best_y,
            xi=xi
        )
        all_figures.append(fig_final)
        
        if verbose:
            print(f"\nCreated {len(all_figures)} plots total")
            print("Close all plot windows to continue...")
        
        plt.show()  # Keep all plots open until user closes them

    return X_observed, y_observed, best_x, best_y


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Define a test objective function (black-box)
    def objective_function(x):
        """
        Example objective: 
        f(x) = -(x-2)² sin(5x) + 5

        This function has multiple local maxima and one global maximum.
        """
        return -(x - 2)**2 * np.sin(5 * x) + 5

    # Set random seed for reproducibility
    np.random.seed(42)

    # Define search space
    bounds = (-2.0, 4.0)

    # Run Bayesian Optimization
    X_obs, y_obs, best_x, best_y = bayesian_optimization(
        objective_func=objective_function,
        bounds=bounds,
        n_initial=3,
        n_iterations=100,
        xi=0.01,
        verbose=True,
        plot=True,  # Enable visualization
        plot_interval=10,  # Create a new plot every 10 iterations
        convergence_threshold=1.0e-3,  # Stop if improvement < 0.001
        acq_type='expected_improvement'  # Change to desired acquisition function
    )

    # Compare with true optimum (only possible because we know the function)
    x_true = np.linspace(bounds[0], bounds[1], 10000)
    y_true = objective_function(x_true)
    true_opt_idx = np.argmax(y_true)
    true_opt_x = x_true[true_opt_idx]
    true_opt_y = y_true[true_opt_idx]

    print(f"\nTrue optimum: x* = {true_opt_x:.6f}, f(x*) = {true_opt_y:.6f}")
    print(f"Error: {abs(best_x - true_opt_x):.6f}")
    print(f"\nSuccess! Found optimum within {abs(best_y - true_opt_y):.6f} of true value.")
