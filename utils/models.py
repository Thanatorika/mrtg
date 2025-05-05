import torch
from torch_topological.nn import SignatureLoss
from torch_topological.nn import VietorisRipsComplex

class Autoencoder(torch.nn.Module):
    """Simple linear autoencoder class.

    This module performs simple embeddings based on an MSE loss. This is
    similar to ordinary principal component analysis. Notice that the
    class is only meant to provide a simple example that can be run
    easily even without the availability of a GPU. In practice, there
    are many more architectures with improved expressive power
    available.
    """

    def __init__(self, input_dim, latent_dim=2):
        """Create new autoencoder with pre-defined latent dimension."""
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.latent_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.latent_dim, self.latent_dim),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, self.latent_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.latent_dim, self.input_dim),
            torch.nn.Sigmoid()
        )

        self.loss_fn = torch.nn.MSELoss()

    def encode(self, x):
        """Embed data in latent space."""
        return self.encoder(x)

    def decode(self, z):
        """Decode data from latent space."""
        return self.decoder(z)

    def forward(self, x):
        """Embeds and reconstructs data, returning a loss."""
        z = self.encode(x)
        x_hat = self.decode(z)

        reconstruction_error = self.loss_fn(x, x_hat)
        return reconstruction_error

class TopologicalAutoencoder(torch.nn.Module):
    """Wrapper for a topologically-regularised autoencoder.

    This class uses another autoencoder model and imbues it with an
    additional topology-based loss term.
    """
    def __init__(self, model, lam_recon, lam_topo, loss=SignatureLoss(), vr=VietorisRipsComplex()):
        super().__init__()

        self.lam_recon = lam_recon
        self.lam_topo = lam_topo
        self.model = model
        self.loss = loss
        self.vr = vr

    def forward(self, x):
        z = self.model.encode(x)

        pi_x = self.vr(x)
        pi_z = self.vr(z)

        recon_loss = self.model(x)
        topo_loss = self.loss([x, pi_x], [z, pi_z])

        loss = self.lam_recon * recon_loss + self.lam_topo * topo_loss
        return loss
    
def relaxed_distortion_measure(func, x, eta=0.2, create_graph=True):
    '''
    func: encoder
    '''
    bs = len(x)
    x_perm = x[torch.randperm(bs)]
    alpha = (torch.rand(bs) * (1 + 2*eta) - eta).unsqueeze(1).to(x)
    x_augmented = alpha*x + (1-alpha)*x_perm
    v = torch.randn(x.size()).to(x)
    Jv = torch.autograd.functional.jvp(
        func, x_augmented, v=v, create_graph=create_graph)[1]
    TrG = torch.sum(Jv.view(bs, -1)**2, dim=1).mean()
    JTJv = (torch.autograd.functional.vjp(
        func, x_augmented, v=Jv, create_graph=create_graph)[1]).view(bs, -1)
    TrG2 = torch.sum(JTJv**2, dim=1).mean()
    return TrG2/TrG**2


class IsometricAutoencoder(torch.nn.Module):
    """Wrapper for a geometrically-regularised autoencoder.

    This class uses another autoencoder model and imbues it with an
    additional geometry-based loss term.
    """
    def __init__(self, model, lam_recon, lam_topo, lam_iso, topo_loss=SignatureLoss(), vr=VietorisRipsComplex()):
        super().__init__()

        self.lam_recon = lam_recon
        self.lam_topo = lam_topo
        self.lam_iso = lam_iso

        self.model = model
        self.topo_loss = topo_loss
        self.vr = vr


    def forward(self, x):
        z = self.model.encode(x)

        pi_x = self.vr(x)
        pi_z = self.vr(z)

        recon_loss = self.model(x)
        topo_loss = self.topo_loss([x, pi_x], [z, pi_z])
        iso_loss = relaxed_distortion_measure(func=self.model.encoder, x=x)

        loss = self.lam_recon * recon_loss + self.lam_topo * topo_loss + self.lam_iso * iso_loss
        return loss
    
class ManifoldReconstructionLayer(torch.nn.Module):
    """
        Manifold fitting implementation with pytorch
    """
    def __init__(self, input_dim, r0=0.05, r1=0.01, r2=0.05, k=3.0):
        """Initialize the MRL class.
        Args:
            input_dim: torch.Tensor, shape (n, p, ...), sample data tensor
            r0: float, the radius for calculating the contraction direction, in the order of O(sigma * log(1/sigma)))
            r1: float, the first radius for constructing the contraction region, in the order of O(sigma)
            r2: float, the second radius for calculating the contraction region, in the order of O(sigma * log(1/sigma)))
            k: float, the power parameter for calculating the weight
        """
        super(ManifoldReconstructionLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = input_dim

        self.r0_init = r0
        self.r1_init = r1
        self.r2_init = r2
        self.r0 = torch.nn.Parameter(torch.tensor(r0), requires_grad=True)
        self.r1 = torch.nn.Parameter(torch.tensor(r1), requires_grad=True)
        self.r2 = torch.nn.Parameter(torch.tensor(r2), requires_grad=True)
        self.k = k
    
    def weight1(self, dist, r):
        w = torch.zeros_like(dist)
        # check whether the radius is NA
        if r.isnan():
            w = torch.ones_like(dist)
        else:
            flag1 = dist < r
            w[flag1] = (1 - (dist[flag1]/r)**2)**self.k

        return w
    
    def weight2(self, dist, r):
        w = torch.zeros_like(dist)  # Initialize tensor w with zeros

        flag1 = dist < r / 2
        flag2 = (dist >= r / 2) & (dist < r)

        w[flag1] = 1.0
        w[flag2] = (1 - ((2 * dist[flag2] - r) / r) ** 2) ** self.k

        return w
    
    def Normalize_weight(self, w):
        return w / torch.sum(w, dim=0, keepdim=True)

    def forward(self, x):
        """Compute the manifold fitting function G(x).
        Args:
            x: torch.Tensor, shape (n, p, ...), new data tensor
        """

        y = x.clone()
        dist = torch.cdist(x, y) # Compute the distance between input_dim and x, shape (N, n)
        max_dist = dist.max()

        if self.r0 > max_dist:
            self.r0 = torch.nn.Parameter(max_dist.clone(), requires_grad=True)
        if self.r1 > max_dist:
            self.r1 = torch.nn.Parameter(max_dist.clone(), requires_grad=True)
        if self.r2 > max_dist:
            self.r2 = torch.nn.Parameter(max_dist.clone(), requires_grad=True)

        inds = torch.any(dist <= self.r0, dim=1).nonzero().squeeze() # Find the indices of input_dim that are within the contraction region of x
        dist = dist[inds] # Select the distance between input_dim and x within the contraction region, shape (N, n)
        x = x[inds] # Select the data within the contraction region, shape (N, p)
        alpha = self.weight1(dist, self.r0)
        alpha = self.Normalize_weight(alpha)

        # Compute mu: the F(y) function for each y
        mu = torch.matmul(alpha.t(), x)
        flag0 = torch.isnan(mu)
        mu[flag0] = y[flag0] # If all weights are zero, set mu as x

        U = y - mu 
        U = torch.unsqueeze(U, dim=1) # Add a dimension to U to have shape (n, 1, p)
        # UTU = torch.div(torch.matmul(U.transpose(1, 2),U), torch.matmul(U,U.transpose(1, 2))) # Compute UU^T

        diff_vectors = torch.unsqueeze(x, dim=1) - torch.unsqueeze(y, dim=0) # Compute input_dim - x, shape (N, n, p)
        diff_vectors = torch.unsqueeze(diff_vectors, dim=2) # Add a dimension to have shape (N, n, 1, p)
        # projection = torch.matmul(diff_vectors, UTU) # Compute the projection of input_dim - x onto U, shape (N, n, 1, p)
        projection = torch.matmul(diff_vectors, U.transpose(1, 2)) # Compute the projection of input_dim - x onto U, shape (N, n, 1, 1)
        projection = torch.matmul(projection, U) # Compute the projection of input_dim - x onto U, shape (N, n, 1, p)

        dist_u = torch.matmul(projection, projection.transpose(2, 3)) # Compute the squared distance between input_dim and x along U, shape (N, n, 1, 1)
        dist_u = torch.squeeze(dist_u).clamp_min(1e-6) # Squeeze dist_u to have shape (N, n)
        dist_v = (dist**2 - dist_u).clamp_min(1e-6)
        dist_v = dist_v**0.5 # Compute the distance between input_dim and x along V, shape (N, n)
        dist_u = dist_u**0.5 # Compute the distance between input_dim and x along U, shape (N, n)

        beta = self.weight2(dist_v, self.r1) * self.weight2(dist_u, self.r2)
        beta = self.Normalize_weight(beta)

        # Compute the manifold fitting function G(x) for each y
        e_Z = torch.matmul(beta.t(), x)
        flag0 = torch.isnan(e_Z)
        e_Z[flag0] = y[flag0] # If all weights are zero, set e as y


        return e_Z
    
class MRTGAutoencoder(torch.nn.Module):
    def __init__(self, input_dim, latent_dim=2, r0=0.05, r1=0.01, r2=0.05, k=3.0,
                 lam_recon=1, lam_topo=0, lam_geom=0, ae_loss=torch.nn.MSELoss(), topo_reg=SignatureLoss(), vr=VietorisRipsComplex()):

        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.r0 = r0
        self.r1 = r1
        self.r2 = r2
        self.k = k

        self.lam_recon = lam_recon
        self.lam_topo = lam_topo
        self.lam_geom = lam_geom
        
        self.recon_loss = ae_loss
        self.topo_reg = topo_reg
        self.vr = vr

        self.mr_layer = ManifoldReconstructionLayer(input_dim=self.input_dim, r0=self.r0, r1=self.r1, r2=self.r2, k=self.k)

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.latent_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.latent_dim, self.latent_dim),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, self.latent_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.latent_dim, self.input_dim),
            torch.nn.Sigmoid()
        )

    def mu(self, x):
        return self.mr_layer(x)

    def encode(self, x):
        """Embed data in latent space."""
        return self.encoder(x)

    def decode(self, z):
        """Decode data from latent space."""
        return self.decoder(z)

    def forward(self, x):
        """Embeds and reconstructs data, returning a loss."""
        y = self.mu(x)
        z = self.encode(y)
        x_hat = self.decode(z)

        pi_y = self.vr(y)
        pi_z = self.vr(z)

        self.ae_error = self.recon_loss(x, x_hat)
        self.topo_error = self.topo_reg([y, pi_y], [z, pi_z])
        self.geom_error = relaxed_distortion_measure(func=self.encoder, x=y)

        return self.lam_recon * self.ae_error + self.lam_topo * self.topo_error + self.lam_geom * self.geom_error