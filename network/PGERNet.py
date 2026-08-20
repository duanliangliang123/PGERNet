import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile


def get_adaptive_groups(channels):
    # Determine valid GroupNorm group count based on channel divisibility
    return 8 if channels % 8 == 0 else (4 if channels % 4 == 0 else 2)


class SmoothingResidualBlock(nn.Module):
    # Basic 3x3 residual block with GroupNorm and SiLU activation
    def __init__(self, dim):
        super().__init__()
        groups = get_adaptive_groups(dim)
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GroupNorm(groups, dim),
            nn.SiLU()
        )

    def forward(self, x):
        return self.block(x) + x


class EntropyPool2d(nn.Module):
    # Computes normalized spatial Shannon entropy per channel
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        x_flat = x.view(B, C, -1)
        p = F.softmax(x_flat, dim=-1)

        entropy = -(p * torch.log(p + self.eps)).sum(dim=-1)
        max_entropy = torch.log(torch.tensor(N, dtype=x.dtype, device=x.device))

        normalized_entropy = entropy / (max_entropy + self.eps)
        return normalized_entropy.view(B, C, 1, 1)


class EntropyDrivenElasticRefinement(nn.Module):
    # Refine features using isotropic diffusion, anisotropic shear wave kernels, and entropy attention
    def __init__(self, in_dim, out_dim, reduction=4, wave_scales=(5, 9, 13), use_entropy=True):
        super().__init__()
        self.use_entropy = use_entropy
        self.proj = nn.Conv2d(in_dim, out_dim, 1) if in_dim != out_dim else None
        gn_groups = get_adaptive_groups(out_dim)

        # Local isotropic diffusion pathway
        self.isotropic_diffusion = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 1, bias=False),
            nn.GroupNorm(gn_groups, out_dim), nn.SiLU(),
            nn.Conv2d(out_dim, out_dim, 3, padding=1, bias=False),
            nn.GroupNorm(gn_groups, out_dim), nn.SiLU()
        )

        # Directional shear wave propagation pathways
        self.shear_wave_propagators = nn.ModuleList([
            nn.ModuleDict({
                'v': nn.Conv2d(out_dim, out_dim // 2, kernel_size=(k, 1), padding=(k // 2, 0), bias=False),
                'h': nn.Conv2d(out_dim, out_dim // 2, kernel_size=(1, k), padding=(0, k // 2), bias=False)
            }) for k in wave_scales
        ])

        self.wave_fusion = nn.Sequential(
            nn.Conv2d(out_dim // 2 * len(wave_scales), out_dim, 1, bias=False),
            nn.GroupNorm(gn_groups, out_dim), nn.SiLU()
        )

        self.elastic_field_fusion = nn.Sequential(
            nn.Conv2d(out_dim * 2, out_dim, 1, bias=False),
            nn.GroupNorm(gn_groups, out_dim), nn.SiLU()
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.entropy_pool = EntropyPool2d()

        # Energy dissipation channel attention
        self.energy_dissipation = nn.Sequential(
            nn.Conv2d(out_dim, out_dim // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim // reduction, out_dim, 1, bias=False),
            nn.Sigmoid()
        )

        self.final_equilibrium = nn.Conv2d(out_dim, out_dim, 1)

    def forward(self, x):
        if self.proj is not None:
            x = self.proj(x)

        field_iso = self.isotropic_diffusion(x)

        # Aggregate multi-scale directional response
        wave_responses = [F.silu(prop['v'](x) + prop['h'](x)) for prop in self.shear_wave_propagators]
        field_aniso = self.wave_fusion(torch.cat(wave_responses, dim=1))

        combined_field = self.elastic_field_fusion(torch.cat([field_iso, field_aniso], dim=1))

        # Pool field energy via spatial entropy or global average pooling
        if self.use_entropy:
            pooled_energy = self.entropy_pool(combined_field)
        else:
            pooled_energy = self.avg_pool(combined_field)

        attention_weights = self.energy_dissipation(pooled_energy)
        equilibrium_state = self.final_equilibrium(combined_field * attention_weights)

        return equilibrium_state + x


class EnhancedResidualBlock(nn.Module):
    # Convolutional residual block enhanced with elastic refinement
    def __init__(self, in_dim, out_dim, use_entropy=True):
        super().__init__()
        groups = get_adaptive_groups(out_dim)
        self.block = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1, bias=False),
            nn.GroupNorm(groups, out_dim),
            nn.SiLU(),
            nn.Conv2d(out_dim, out_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_dim),
            nn.SiLU(),
            EntropyDrivenElasticRefinement(in_dim=out_dim, out_dim=out_dim, use_entropy=use_entropy)
        )
        self.shortcut = nn.Conv2d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        return self.block(x) + self.shortcut(x)


class Encoder(nn.Module):
    # Hierarchical feature extraction encoder with downsampling stages
    def __init__(self, dim, dim_mults, num_blocks=1, entropy_stages=2):
        super().__init__()
        self.dims = [dim * m for m in dim_mults]
        in_out = list(zip([dim] + self.dims[:-1], self.dims))

        if isinstance(num_blocks, int):
            num_blocks = [num_blocks] * len(in_out)

        self.blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for idx, ((in_dim, out_dim), n) in enumerate(zip(in_out, num_blocks)):
            use_ent = (idx < entropy_stages)
            layers = [EnhancedResidualBlock(in_dim if i == 0 else out_dim, out_dim, use_entropy=use_ent) for i in
                      range(n)]
            self.blocks.append(nn.Sequential(*layers))
            self.downsamples.append(nn.Conv2d(out_dim, out_dim, 3, stride=2, padding=1))

    def forward(self, x):
        skips = []
        for block, downsample in zip(self.blocks, self.downsamples):
            x = block(x)
            skips.append(x)
            x = downsample(x)
        return skips, x


class Decoder(nn.Module):
    # Progressive upsampling decoder with cross-scale skip connections
    def __init__(self, dim, dim_mults, num_blocks=1, entropy_stages=2):
        super().__init__()
        encoder_dims = [dim * m for m in dim_mults]
        reversed_dims = list(reversed(encoder_dims))
        reversed_dims.append(dim)
        in_out = list(zip(reversed_dims, reversed_dims[1:]))

        if isinstance(num_blocks, int):
            num_blocks = [num_blocks] * len(in_out)

        self.upsamples = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.feature_processing_blocks = nn.ModuleList()

        total_stages = len(in_out)
        for idx, ((in_dim, out_dim), n) in enumerate(zip(in_out, num_blocks)):
            use_ent = (idx >= total_stages - entropy_stages)
            self.upsamples.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_dim, in_dim, 3, padding=1)
            ))
            layers = [EnhancedResidualBlock(in_dim + in_dim if i == 0 else out_dim, out_dim, use_entropy=use_ent)
                      for i in range(n)]
            self.blocks.append(nn.Sequential(*layers))
            self.feature_processing_blocks.append(SmoothingResidualBlock(out_dim))

    def forward(self, x, fused_skips):
        for upsample, block, feature_block, skip in zip(self.upsamples, self.blocks, self.feature_processing_blocks,
                                                        reversed(fused_skips)):
            x = upsample(x)
            x = torch.cat([x, skip], dim=1)
            x = block(x)
            x = feature_block(x)
        return x


class DynamicRoutingAlignmentUnit(nn.Module):
    # Dynamically routes and aligns cross-modal features via uniform and non-uniform expert modules
    def __init__(self, dim, reduction=4):
        super().__init__()
        groups = get_adaptive_groups(dim)
        diff_dim = max(1, dim // 2)

        self.stem = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=False),
            nn.GroupNorm(groups, dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, dim),
            nn.SiLU(),
        )

        self.align_head_non_uniform = nn.Conv2d(dim, dim * 4, 3, padding=1, bias=False)
        self.align_head_uniform = nn.Conv2d(dim, dim * 4, 3, padding=1, bias=False)

        self.diff_embed = nn.Sequential(
            nn.Conv2d(dim, diff_dim, 1, bias=False),
            nn.SiLU(),
        )

        post_dim = dim + diff_dim
        self.post_non_uniform_a = nn.Sequential(
            nn.Conv2d(post_dim, dim, 3, padding=1, bias=False), nn.GroupNorm(groups, dim))
        self.post_non_uniform_b = nn.Sequential(
            nn.Conv2d(post_dim, dim, 3, padding=1, bias=False), nn.GroupNorm(groups, dim))

        self.post_uniform_a = nn.Sequential(
            nn.Conv2d(post_dim, dim, 3, padding=1, bias=False), nn.GroupNorm(groups, dim), nn.SiLU())
        self.post_uniform_b = nn.Sequential(
            nn.Conv2d(post_dim, dim, 3, padding=1, bias=False), nn.GroupNorm(groups, dim), nn.SiLU())

        # Spatial-channel routing mechanism
        mid = max(4, dim // reduction)
        self.router_local = nn.Sequential(
            nn.Conv2d(dim, mid, 3, padding=1, bias=False), nn.SiLU(),
            nn.Conv2d(mid, mid, 3, padding=1, bias=False), nn.SiLU(),
        )
        self.router_global = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, mid, 1, bias=False), nn.SiLU(),
        )
        self.router_fuse = nn.Sequential(
            nn.Conv2d(mid * 2, mid, 1, bias=False), nn.SiLU(),
            nn.Conv2d(mid, 1, 1, bias=False), nn.Sigmoid(),
        )

    def _route(self, f):
        local = self.router_local(f)
        g = self.router_global(f).expand(-1, -1, *f.shape[-2:])
        return self.router_fuse(torch.cat([local, g], dim=1))

    def _align(self, a, b, head, gate_only=False):
        g_a, m_a, g_b, m_b = torch.chunk(head, 4, dim=1)
        if gate_only:
            m_a, m_b = F.silu(m_a), F.silu(m_b)
        a_aligned = a + torch.sigmoid(g_a) * m_a
        b_aligned = b + torch.sigmoid(g_b) * m_b
        return a_aligned, b_aligned

    def forward(self, a, b):
        f = self.stem(torch.cat([a, b], dim=1))
        alpha = self._route(f)

        a_nu, b_nu = self._align(a, b, self.align_head_non_uniform(f), gate_only=False)
        a_un, b_un = self._align(a, b, self.align_head_uniform(f), gate_only=True)

        diff_nu = self.diff_embed(torch.abs(a_nu - b_nu))
        a_out_nu = self.post_non_uniform_a(torch.cat([a_nu, diff_nu], dim=1)) + a_nu
        b_out_nu = self.post_non_uniform_b(torch.cat([b_nu, diff_nu], dim=1)) + b_nu

        diff_un = self.diff_embed(torch.abs(a - b))
        a_out_un = self.post_uniform_a(torch.cat([a_un, diff_un], dim=1)) + a_un
        b_out_un = self.post_uniform_b(torch.cat([b_un, diff_un], dim=1)) + b_un

        # Blend non-uniform and uniform outputs using routing weights
        a_out = alpha * a_out_nu + (1 - alpha) * a_out_un
        b_out = alpha * b_out_nu + (1 - alpha) * b_out_un

        return a_out, b_out


class FixedGaussianBlur(nn.Module):
    # Non-trainable 3x3 Gaussian spatial smoothing filter
    def __init__(self, channels):
        super().__init__()
        kernel = torch.tensor([[1., 2., 1.],
                               [2., 4., 2.],
                               [1., 2., 1.]]) / 16.0
        weight = kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        self.register_buffer('weight', weight)
        self.groups = channels

    def forward(self, x):
        return F.conv2d(x, self.weight, padding=1, groups=self.groups)


class GuidanceEstimator(nn.Module):
    # Predicts guidance features and spatial confidence maps
    def __init__(self, channels):
        super().__init__()
        groups = 8 if channels >= 8 else 1
        self.feat = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        self.conf = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        f = self.feat(x)
        c = self.conf(f)
        return f, c


class ConcatConvFusion(nn.Module):
    # Fuses two feature representations via concatenation and residual convolutions
    def __init__(self, channels):
        super().__init__()
        groups = 8 if channels >= 8 else 1
        self.conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, featA, featB):
        x = torch.cat([featA, featB], dim=1)
        out = self.conv(x)
        return out + featA


class RecHead(nn.Module):
    # Auxiliary 3-channel reconstruction head for deep supervision
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // 2, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_dim // 2, 3, 1)
        )

    def forward(self, x):
        return self.net(x)


class ProgressiveGuidanceCrossFusion(nn.Module):
    # Frequency-decomposed cross-modal fusion guided by uncertainty maps and optional prior features
    def __init__(self, channels, prior_dim=None):
        super().__init__()
        self.channels = channels
        self.use_prior = prior_dim is not None

        if self.use_prior:
            self.prior_proj = nn.Sequential(
                nn.Conv2d(prior_dim, channels, 1),
                nn.GroupNorm(get_adaptive_groups(channels), channels),
                nn.SiLU()
            )
            self.film = nn.Conv2d(channels, channels * 2, 1)

        self.rec_head = RecHead(channels)

        self.nir_to_t = GuidanceEstimator(channels)
        self.rgb_to_A = GuidanceEstimator(channels)

        self.lowpass_rgb = FixedGaussianBlur(channels)
        self.lowpass_nir = FixedGaussianBlur(channels)

        self.ca_rgb = ConcatConvFusion(channels)
        self.ca_nir = ConcatConvFusion(channels)

        groups = 8 if channels >= 8 else 1
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU()
        )
        self.ugate = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def _guided_feature_extraction(self, F_rgb, F_nir):
        g_t, c_t = self.nir_to_t(F_nir)
        g_A, c_A = self.rgb_to_A(F_rgb)
        return g_t, c_t, g_A, c_A

    def _frequency_aware_cross_fusion(self, F_rgb, F_nir, g_t, g_A, c_t, c_A):
        # High/Low frequency band splitting for RGB
        rgb_low = self.lowpass_rgb(F_rgb)
        rgb_high = F_rgb - rgb_low
        t_low = self.lowpass_nir(g_t)
        t_high = g_t - t_low

        # High/Low frequency band splitting for NIR
        nir_low = self.lowpass_nir(F_nir)
        nir_high = F_nir - nir_low
        A_low = self.lowpass_rgb(g_A)
        A_high = g_A - A_low

        # Confidence-weighted frequency aggregation
        rgb_att_high = self.ca_rgb(rgb_high, t_high)
        rgb_att_low = self.ca_rgb(rgb_low, t_low)
        rgb_att = c_t * rgb_att_high + (1.0 - c_t) * rgb_att_low

        nir_att_high = self.ca_nir(nir_high, A_high)
        nir_att_low = self.ca_nir(nir_low, A_low)
        nir_att = c_A * nir_att_high + (1.0 - c_A) * nir_att_low

        return rgb_att, nir_att

    def _uncertainty_guided_fusion(self, F_rgb, F_nir, rgb_att, nir_att, g_t, g_A):
        gate = self.ugate(torch.cat([torch.abs(F_rgb - F_nir), g_t, g_A], dim=1))
        fused = self.fuse(torch.cat([rgb_att, nir_att], dim=1))
        base = 0.5 * (F_rgb + F_nir)
        out = gate * fused + (1.0 - gate) * base
        return out

    def forward(self, F_rgb, F_nir, prior=None):
        # Modulate feature maps using prior information via FiLM layers
        if self.use_prior and prior is not None:
            if prior.shape[-1] != F_rgb.shape[-1]:
                prior_up = F.interpolate(prior, size=F_rgb.shape[2:], mode='bilinear', align_corners=False)
            else:
                prior_up = prior
            P = self.prior_proj(prior_up)
            gamma, beta = self.film(P).chunk(2, dim=1)
            F_rgb = F_rgb * (1 + gamma) + beta
            F_nir = F_nir * (1 + gamma) + beta

        g_t, c_t, g_A, c_A = self._guided_feature_extraction(F_rgb, F_nir)
        rgb_att, nir_att = self._frequency_aware_cross_fusion(F_rgb, F_nir, g_t, g_A, c_t, c_A)
        out = self._uncertainty_guided_fusion(F_rgb, F_nir, rgb_att, nir_att, g_t, g_A)

        return out, self.rec_head(out)


class PGERNet(nn.Module):
    # Dual-branch RGB-NIR restoration network featuring dynamic alignment and progressive fusion
    def __init__(self, dim=32, dim_mults=(1, 2, 4, 8), num_blocks_encoder=1, num_blocks_decoder=1, entropy_stages=2):
        super().__init__()
        self.init_conv_rgb = nn.Conv2d(3, dim, 7, padding=3)
        self.encoder_rgb = Encoder(dim, dim_mults, num_blocks=num_blocks_encoder, entropy_stages=entropy_stages)

        self.init_conv_nir = nn.Conv2d(1, dim, 7, padding=3)
        self.encoder_nir = Encoder(dim, dim_mults, num_blocks=num_blocks_encoder, entropy_stages=entropy_stages)

        encoder_dims = [dim * m for m in dim_mults]
        mid_dim = encoder_dims[-1]

        self.align_mid = DynamicRoutingAlignmentUnit(mid_dim)
        self.pgca_mid = ProgressiveGuidanceCrossFusion(mid_dim, prior_dim=None)

        self.mid_block = nn.Sequential(
            EnhancedResidualBlock(mid_dim, mid_dim, use_entropy=False),
            EnhancedResidualBlock(mid_dim, mid_dim, use_entropy=False)
        )

        reversed_dims = list(reversed(encoder_dims))
        prior_dims = [mid_dim] + reversed_dims[:-1]

        self.skip_fusions = nn.ModuleList()
        self.skip_alignments = nn.ModuleList()

        for i in range(len(reversed_dims)):
            self.skip_fusions.append(
                ProgressiveGuidanceCrossFusion(channels=reversed_dims[i], prior_dim=prior_dims[i])
            )

            self.skip_alignments.append(
                DynamicRoutingAlignmentUnit(dim=reversed_dims[i])
            )

        self.decoder = Decoder(dim, dim_mults, num_blocks=num_blocks_decoder, entropy_stages=entropy_stages)
        self.final_conv = nn.Conv2d(dim, 3, 3, padding=1)

    def forward(self, rgb, nir):
        # Extract features along encoder streams
        x_rgb = self.init_conv_rgb(rgb)
        skips_rgb, x_rgb = self.encoder_rgb(x_rgb)

        x_nir = self.init_conv_nir(nir)
        skips_nir, x_nir = self.encoder_nir(x_nir)

        # Bottleneck alignment, fusion, and processing
        x_rgb_aligned, x_nir_aligned = self.align_mid(x_rgb, x_nir)
        x, y_mid = self.pgca_mid(x_rgb_aligned, x_nir_aligned)
        x = self.mid_block(x)

        fused_skips_deep_to_shallow = []
        aux_preds = [y_mid]
        prior = x

        # Process skip connections from deepest to shallowest stage
        for i in range(len(skips_rgb)):
            idx = -(i + 1)
            s_rgb = skips_rgb[idx]
            s_nir = skips_nir[idx]

            s_rgb_aligned, s_nir_aligned = self.skip_alignments[i](s_rgb, s_nir)
            s_fused, y_skip = self.skip_fusions[i](s_rgb_aligned, s_nir_aligned, prior=prior)
            fused_skips_deep_to_shallow.append(s_fused)
            aux_preds.append(y_skip)

            prior = s_fused

        fused_skips = list(reversed(fused_skips_deep_to_shallow))

        # Decode features into output image
        x = self.decoder(x, fused_skips)
        out = self.final_conv(x)

        if self.training:
            return out, aux_preds

        return out


def calculate_flops_params():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 1. Initialize model and inputs
    model = PGERNet(dim=16, dim_mults=(1, 2, 4, 5),
                    num_blocks_encoder=[1, 1, 1, 1],
                    num_blocks_decoder=[1, 1, 1, 1]).to(device)
    
    # Enable cudnn.benchmark for fixed input sizes
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True

    rgb_input = torch.randn(1, 3, 256, 256).to(device)
    nir_input = torch.randn(1, 1, 256, 256).to(device)

    model.eval()

    # 2. Benchmark inference time
    warmup_iters = 50
    measure_iters = 100
    
    with torch.no_grad():
        print(f"Starting warm-up for {warmup_iters} iterations...")
        # Warm-up phase
        for _ in range(warmup_iters):
            _ = model(rgb_input, nir_input)
            
        print(f"Measuring inference time for {measure_iters} iterations...")
        # Measurement phase
        if device == 'cuda':
            # Synchronize before timing
            torch.cuda.synchronize()
            
            # High-precision timing with CUDA Events
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            
            starter.record()
            for _ in range(measure_iters):
                result = model(rgb_input, nir_input)
            ender.record()
            
            # Wait for GPU tasks to complete
            torch.cuda.synchronize()
            
            # Calculate average time in ms
            total_time_ms = starter.elapsed_time(ender)
            avg_time_ms = total_time_ms / measure_iters
            
        else:
            # High-precision timing for CPU
            start = time.perf_counter()
            for _ in range(measure_iters):
                result = model(rgb_input, nir_input)
            end = time.perf_counter()
            
            total_time_ms = (end - start) * 1000
            avg_time_ms = total_time_ms / measure_iters

        print(f"Average Forward pass time: {avg_time_ms:.2f} ms")
        print(f"FPS: {1000 / avg_time_ms:.2f}")
        print("Output shape:", tuple(result.shape))

    # 3. Calculate FLOPs and Params
    # Move to CPU to prevent potential profiling errors
    model.to('cpu')
    rgb_input = rgb_input.to('cpu')
    nir_input = nir_input.to('cpu')

    try:
        flops, params = profile(model, inputs=(rgb_input, nir_input), verbose=False)
        gflops = flops / 1e9
        params_m = params / 1e6
        print(f"FLOPs: {gflops:.2f} G")
        print(f"Params: {params_m:.2f} M")
    except Exception as e:
        print(f"Could not calculate FLOPs and Params: {e}")

if __name__ == "__main__":
    calculate_flops_params()
