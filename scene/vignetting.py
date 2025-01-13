import torch

def create_differentiable_vignetting_mask(image_tensor, scaling_factors):
    _, width, height = image_tensor.shape  # Assume image_tensor is of shape (C, H, W)

    x = torch.arange(width).float().to(image_tensor.device) - (width - 1) / 2
    y = torch.arange(height).float().to(image_tensor.device) - (height - 1) / 2
    X, Y = torch.meshgrid(x, y, indexing="ij")

    distances = torch.sqrt(X**2 + Y**2)

    max_distance = distances.max()
    distances_normalized = distances / max_distance * (scaling_factors.size(0) - 1)

    lower_indices = distances_normalized.floor().long().clamp(0, scaling_factors.size(0) - 2)
    upper_indices = lower_indices + 1

    lower_weights = (upper_indices - distances_normalized).cuda()  # weight for lower index
    upper_weights = (distances_normalized - lower_indices).cuda()  # weight for upper index

    # Get the scaling factors for the lower and upper indices
    scaling_mask = lower_weights * scaling_factors[lower_indices] + upper_weights * scaling_factors[upper_indices]

    return scaling_mask

class VignettingModel(torch.nn.Module):
    def __init__(self, n_terms=4, device='cuda'):
        super(VignettingModel, self).__init__()

        # Number of terms in the summation (higher order model)
        self.n_terms = n_terms

        # Initialize learnable parameters (requires_grad=True for optimization)
        # Initialize a_k close to zero for minimal vignetting effect at the start
        self.a_k = torch.nn.Parameter(torch.full((n_terms,), 0.01, device=device))  # Coefficients a_k initialized near 0

        # Initialize beta_k to a small value but ensuring it gives a smooth falloff
        self.beta_k = torch.nn.Parameter(torch.linspace(2.0, 8.0, n_terms).to(device))

        # Initialize gamma to 1 for neutral global exponentiation
        #self.gamma = torch.nn.Parameter(torch.tensor(1.0).to(device))
        self.gamma = 1.

        # Store the device (e.g., 'cuda' or 'cpu')
        self.device = device

    def bak_forward(self, image_size):
        # Get image dimensions
        height, width = image_size

        # Compute the image center
        x_c, y_c = width / 2, height / 2

        # Compute the maximum possible distance from the center (diagonal distance)
        r_max = torch.sqrt(torch.tensor((x_c ** 2 + y_c ** 2), dtype=torch.float32).to(self.device))

        # Create a meshgrid of pixel coordinates on the specified device
        y, x = torch.meshgrid(torch.arange(height, dtype=torch.float32).to(self.device),
                              torch.arange(width, dtype=torch.float32).to(self.device))

        # Calculate the distance of each pixel from the image center
        d = torch.sqrt((x - x_c) ** 2 + (y - y_c) ** 2)

        # Normalize the distance by r_max
        d_normalized = d / r_max

        # Initialize the vignetting mask
        mask = torch.zeros_like(d_normalized)

        # Sum over the higher-order terms to compute the vignetting mask
        for k in range(self.n_terms):
            mask += self.a_k[k] * d_normalized ** self.beta_k[k]

        # Apply the global exponent gamma
        mask = 1 - mask.clamp(0, 1) ** self.gamma

        return mask

    def forward(self, image_size):
        # Get image dimensions
        height, width = image_size

        # Compute the image center
        x_c, y_c = width / 2, height / 2

        # Create a meshgrid of pixel coordinates on the specified device
        y, x = torch.meshgrid(torch.arange(height, dtype=torch.float32).to(self.device),
                              torch.arange(width, dtype=torch.float32).to(self.device))

        # Calculate the distance of each pixel from the image center
        r = torch.sqrt((x - x_c) ** 2 + (y - y_c) ** 2)

        # Apply the arctan(r)/r normalization
        #r_normalized = torch.arctan(r) / r
        r_normalized = torch.arctan(r)

        # Replace NaNs from division by zero at the center (r=0) with 1
        r_normalized[r == 0] = 1

        # Initialize the vignetting mask
        mask = torch.zeros_like(r_normalized)

        # Sum over the higher-order terms to compute the vignetting mask
        for k in range(self.n_terms):
            mask += self.a_k[k] * r_normalized ** self.beta_k[k]

        # Apply the global exponent gamma
        mask = 1 - mask.clamp(0, 1) ** self.gamma

        return mask
