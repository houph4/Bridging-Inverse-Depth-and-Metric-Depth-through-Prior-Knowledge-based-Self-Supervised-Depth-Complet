import numpy as np
from scipy.optimize import minimize_scalar
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
def compute_scale_and_shift_ls(prediction, target, mask):
    # tuple specifying with axes to sum
    sum_axes = (0, 1)

    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = np.sum(mask * prediction * prediction, sum_axes)
    a_01 = np.sum(mask * prediction, sum_axes)
    a_11 = np.sum(mask, sum_axes)

    # right hand side: b = [b_0, b_1]
    b_0 = np.sum(mask * prediction * target, sum_axes)
    b_1 = np.sum(mask * target, sum_axes)
    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = np.zeros_like(b_0)
    x_1 = np.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    # A needs to be a positive definite matrix.
    valid = det > 0

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def compute_scale_and_shift_theil_sen(prediction, target, mask, sample_size=5000, inlier_threshold=1.5):
    """
    Theil-Sen Estimator for estimating scale and shift in depth prediction tasks with random sub-sampling.

    Parameters:
    - prediction: Predicted depth values as a NumPy array.
    - target: True depth values as a NumPy array.
    - mask: Binary mask indicating valid pixels (1 for valid, 0 for invalid).
    - sample_size: Number of random point pairs to sample for computing the slopes.
    - inlier_threshold: Threshold for considering a point as an inlier based on residuals.
    - visualize: Boolean flag to visualize the results.

    Returns:
    - best_scale: Estimated scale.
    - best_shift: Estimated shift.
    - best_inlier_count: Number of inliers after estimation.
    - inlier_ratio: Ratio of inliers to total valid points.
    """
    # Extract valid indices based on the mask
    valid_indices = np.where((mask > 0) & (target > 0))  # target为real_depth
    prediction_valid = prediction[valid_indices]
    target_valid = target[valid_indices]

    num_valid_points = len(prediction_valid)

    if num_valid_points < 2:
        raise ValueError("Not enough valid points to perform Theil-Sen estimation.")

    # Randomly sample pairs of points for slope calculation
    if sample_size > num_valid_points * (num_valid_points - 1) / 2:
        sample_size = int(num_valid_points * (num_valid_points - 1) / 2)  # Limit to max possible pairs

    # Randomly select point pairs
    sampled_pairs = np.random.choice(num_valid_points, (sample_size, 2), replace=False)

    slopes = []
    intercepts = []
    for i, j in sampled_pairs:
        if prediction_valid[j] != prediction_valid[i]:  # Avoid division by zero
            slope = (target_valid[j] - target_valid[i]) / (prediction_valid[j] - prediction_valid[i])
            intercept = target_valid[i] - slope * prediction_valid[i]
            slopes.append(slope)
            intercepts.append(intercept)

    # Compute the median slope and median intercept (Theil-Sen Estimator)
    best_scale = np.median(slopes)
    best_shift = np.median(intercepts)

    # Create a mask of final inliers based on the estimated scale and shift
    residuals = np.abs(prediction * best_scale + best_shift - target)
    best_mask = (mask * (residuals < inlier_threshold)).astype(np.float32)

    # Calculate inlier count and inlier ratio
    best_inlier_count = np.sum(best_mask)
    inlier_ratio = best_inlier_count / num_valid_points

    # print('best_inlier_count: ', best_inlier_count)
    # print('inlier_ratio: ', inlier_ratio)

    return best_scale, best_shift, inlier_ratio

def compute_scale_and_shift_ransac(prediction, target, mask,
                                   num_iterations, sample_size,
                                   inlier_threshold, inlier_ratio_threshold,
                                   visualize=False):
    best_scale = 0.0
    best_shift = 0.0
    best_inlier_count = 0

    valid_indices = np.where(mask)
    valid_count = len(valid_indices[0])

    for i in range(num_iterations):
        if valid_count < sample_size:
            break

        # Randomly sample from valid indices
        indices = np.random.choice(valid_count, size=sample_size, replace=False)
        mask_sample = np.zeros_like(mask)
        mask_sample[valid_indices[0][indices], valid_indices[1][indices]] = 1

        # Calculate x_0 and x_1 for the sampled data
        sum_axes = (0, 1)
        a_00 = np.sum(mask_sample * prediction * prediction, sum_axes)
        a_01 = np.sum(mask_sample * prediction, sum_axes)
        a_11 = np.sum(mask_sample, sum_axes)
        b_0 = np.sum(mask_sample * prediction * target, sum_axes)
        b_1 = np.sum(mask_sample * target, sum_axes)
        det = a_00 * a_11 - a_01 * a_01
        valid = det > 0
        x_0 = np.zeros_like(b_0)
        x_1 = np.zeros_like(b_1)
        x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
        x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

        # Calculate residuals and count inliers
        residuals = np.abs(mask * prediction * x_0 + x_1 - mask * target)
        residuals = residuals[mask]

        inlier_count = np.sum(residuals < inlier_threshold)

        # Update best model if current model has more inliers
        if inlier_count > best_inlier_count:
            best_scale = x_0
            best_shift = x_1
            best_inlier_count = inlier_count
            inlier_ratio = inlier_count / valid_count
            if inlier_ratio > inlier_ratio_threshold:
                break
    inlier_ratio = best_inlier_count / valid_count

    print('best_inlier_count: ', best_inlier_count)
    print('inlier_ratio: ', best_inlier_count / valid_count)

    if visualize:
    # At the end, visualize the final best mask (inliers) and the original valid area
        best_mask = (mask * (np.abs(prediction * best_scale + best_shift - target) < inlier_threshold)).astype(np.float32)

        # Create custom color maps
        valid_cmap = ListedColormap(['white', 'blue'])  # White for invalid pixels, blue for valid pixels
        inlier_cmap = ListedColormap(['white', 'red'])  # White for non-inliers, red for inliers

        # Visualize the overlap of the original valid area and the inlier mask
        fig, ax = plt.subplots()
        # Original valid area in blue
        ax.imshow(mask, cmap=valid_cmap, alpha=0.5)
        # Final inlier area in red
        ax.imshow(best_mask, cmap=inlier_cmap, alpha=0.5)

        # Add legend
        legend_elements = [
            Patch(facecolor='blue', edgecolor='blue', alpha=0.5, label='Valid Mask'),
            Patch(facecolor='red', edgecolor='red', alpha=0.5, label='Final Inlier Mask')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        plt.title("Overlap of Valid Mask and Final Inlier Mask")
        plt.show()

    return best_scale, best_shift,inlier_ratio



class LeastSquaresEstimator(object):
    def __init__(self, estimate, target, valid):
        self.estimate = estimate
        self.target = target
        self.valid = valid

        # to be computed
        self.scale = 1.0
        self.shift = 0.0
        self.output = None

    def compute_scale_and_shift_ran(self,
                                num_iterations=60, sample_size=5,
                                inlier_threshold=0.02, inlier_ratio_threshold=0.8):
        self.scale, self.shift = compute_scale_and_shift_ransac(self.estimate, self.target, self.valid,
                                                                num_iterations, sample_size,
                                                                inlier_threshold, inlier_ratio_threshold)

    def compute_scale_and_shift(self):
        self.scale, self.shift = compute_scale_and_shift_ls(self.estimate, self.target, self.valid)


    def apply_scale_and_shift(self):
        self.output = self.estimate * self.scale + self.shift

    def clamp_min_max(self, clamp_min=None, clamp_max=None):
        if clamp_min is not None:
            if clamp_min > 0:
                clamp_min_inv = 1.0/clamp_min
                self.output[self.output > clamp_min_inv] = clamp_min_inv
                assert np.max(self.output) <= clamp_min_inv
            else: # divide by zero, so skip
                pass
        if clamp_max is not None:
            clamp_max_inv = 1.0/clamp_max
            self.output[self.output < clamp_max_inv] = clamp_max_inv



def objective_function(x_0, prediction, target, mask):
    # Calculate x_0 * prediction
    x_0_prediction = x_0 * prediction
    # Calculate the error between x_0 * prediction and target, using the mask
    error = np.sum(mask * abs(x_0_prediction - target))
    return error



class Optimizer(object):
    def __init__(self, estimate, target, valid, depth_type):
        self.estimate = estimate
        self.target = target
        self.valid = valid
        self.depth_type = depth_type
        # to be computed
        self.scale = 1.0
        self.output = None

    def optimize_scale(self):
        if self.depth_type == 'inv':
            bounds = (0.0003, 0.01)
        else:
            bounds = (0.5, 1.6) # pos

        # Minimize the objective function using scipy.optimize.minimize_scalar
        result = minimize_scalar(
            objective_function, args=(self.estimate, self.target, self.valid),
            bounds=bounds
        )

        # Extract the optimized x_0 value from the result
        optimized_x_0 = result.x
        self.scale = optimized_x_0

    def apply_scale(self):
        self.output = self.estimate * self.scale

    def clamp_min_max(self, clamp_min=None, clamp_max=None):
        if clamp_min is not None:
            if clamp_min > 0:
                clamp_min_inv = 1.0/clamp_min
                self.output[self.output > clamp_min_inv] = clamp_min_inv
                assert np.max(self.output) <= clamp_min_inv
            else: # divide by zero, so skip
                pass
        if clamp_max is not None:
            clamp_max_inv = 1.0/clamp_max
            self.output[self.output < clamp_max_inv] = clamp_max_inv

    def clamp_min_max_pos(self, clamp_min=None, clamp_max=None):
        if clamp_min is not None:
            if clamp_min >= 0:
                self.output[self.output < clamp_min] = clamp_min
            else:
                pass
        if clamp_max is not None:
            self.output[self.output > clamp_max] = clamp_max


def compute_scale_and_shift_mlesac(prediction, target, mask,
                                   num_iterations=2000, sample_size=10,
                                   inlier_threshold=0.05, sigma=2.0):
    """
    MLESAC implementation with adjustments for better likelihood.

    Parameters:
    prediction: Predicted depth values.
    target: True depth values.
    mask: Mask of valid pixels (1 for valid, 0 for invalid).
    num_iterations: Number of iterations to run MLESAC.
    sample_size: Number of points to sample in each iteration.
    inlier_threshold: Residual threshold to count as an inlier.
    sigma: Standard deviation of Gaussian noise for likelihood calculation.

    Returns:
    best_scale: Estimated scale.
    best_shift: Estimated shift.
    """
    best_scale = 0.0
    best_shift = 0.0
    best_likelihood = -np.inf  # Initialize to a very low likelihood

    valid_indices = np.where(mask)
    valid_count = len(valid_indices[0])

    for _ in range(num_iterations):
        if valid_count < sample_size:
            break

        # Randomly sample from valid indices
        indices = np.random.choice(valid_count, size=sample_size, replace=False)
        mask_sample = np.zeros_like(mask)
        mask_sample[valid_indices[0][indices], valid_indices[1][indices]] = 1

        # Calculate x_0 and x_1 for the sampled data
        sum_axes = (0, 1)
        a_00 = np.sum(mask_sample * prediction * prediction, sum_axes)
        a_01 = np.sum(mask_sample * prediction, sum_axes)
        a_11 = np.sum(mask_sample, sum_axes)
        b_0 = np.sum(mask_sample * prediction * target, sum_axes)
        b_1 = np.sum(mask_sample * target, sum_axes)
        det = a_00 * a_11 - a_01 * a_01
        valid = det > 0
        x_0 = np.zeros_like(b_0)
        x_1 = np.zeros_like(b_1)
        x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
        x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

        # Calculate residuals for all valid points
        residuals = mask * (prediction * x_0 + x_1 - target)
        residuals = residuals[mask]  # Only consider valid pixels

        # Filter inliers based on inlier_threshold
        inliers = residuals[np.abs(residuals) < inlier_threshold]

        # Compute log-likelihood for inliers using Gaussian likelihood
        likelihoods = -0.5 * (inliers / sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi))
        total_likelihood = np.sum(likelihoods)

        # Update the best scale and shift if we get a better likelihood
        if total_likelihood > best_likelihood:
            best_scale = x_0
            best_shift = x_1
            best_likelihood = total_likelihood

    print('best_likelihood: ', best_likelihood)
    return best_scale, best_shift
