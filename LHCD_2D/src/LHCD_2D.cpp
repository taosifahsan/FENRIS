// ────────────────────────────────────────────────────────────────────────────
// Normalization (standardized across all four FENRIS projects)
// ────────────────────────────────────────────────────────────────────────────
// velocity  x = v / v_th   with  v_th = sqrt(2 T_e / m_e)
// time      t -> nu t      with  nu the collision frequency built from THIS
//                          v_th (nu ~ C / v_th^3), so tau_c here is the
//                          collision time of the sqrt(2T/m) convention
// equilibrium               f_M ~ exp(-x^2), single-particle energy T_e x^2
//
// Collision coefficients in these units:
//   speed diffusion  D_c = 1/(4x^3)      (the -0.25 literal below)
//   speed drag       A   = 1/(2x^2)      (the -0.5 literal below)
//   pitch scattering nu_D = (Zi+1)/(4x^3)
// Check: A/D_c = 2x = -d ln exp(-x^2)/dx, so the Maxwellian is stationary.
//
// (Before 2026-08 this project used v_th = sqrt(T/m), equilibrium
// exp(-x_old^2/2), D_c = 1/(2 x_old^3).  The change of variables
// x = x_old/sqrt(2), t -> t/(2 sqrt 2) maps old runs onto new ones exactly;
// only the diffusion literal changed, drag and pitch are invariant.)
// ────────────────────────────────────────────────────────────────────────────
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <asgard.hpp>
#include <asgard_pde.hpp>

// The user-editable initial condition f0(x, theta): lives in input_data/
// because it is an input, even though it is code.  See the header itself for
// the contract.
#include "initial_condition.hpp"

using P = asgard::default_precision;

using term_identity   = asgard::term_identity;
using term_volume     = asgard::term_volume<P>;
using term_div        = asgard::term_div<P>;
using term_grad       = asgard::term_grad<P>;
using term_1d         = asgard::term_1d<P>;
using term_md         = asgard::term_md<P>;
using separable_func = asgard::separable_func<P>;
using term_interp     = asgard::term_interp<P>;

double constexpr PI = asgard::PI;

P smooth_step(P x, P cut, P width)
{
    return P{1} / (P{1} + std::exp(-(x - cut) / width));
}

// LHCD quasilinear diffusion coefficient. The x_parallel window and the
// optional high-speed x cutoff use the same smoothing
// width. It is applied directly to x_parallel=x*cos(theta) at the window edges
// and as a dimensionless width in x/x_max for the radial cutoff.
class LHCDDiffusion {
public:
    P x_parallel_min, x_parallel_max, height, smoothing_width;
    P cut_center, x_max;

    P parallel_window(P x_parallel) const
    {
        if (smoothing_width == P{0})
            return (x_parallel_min < x_parallel &&
                    x_parallel < x_parallel_max) ? P{1} : P{0};
        return smooth_step(x_parallel, x_parallel_min, smoothing_width)
             * smooth_step(x_parallel_max, x_parallel, smoothing_width);
    }

    P x_cutoff(P x) const
    {
        if (cut_center >= P{1})
            return P{1};
        P const x_fraction = x / x_max;
        if (smoothing_width == P{0})
            return x_fraction < cut_center ? P{1} : P{0};
        return P{0.5} * (P{1} - std::tanh(
            (x_fraction - cut_center) / smoothing_width));
    }

    P val(P const &x, P const &theta) const
    {
        return height * parallel_window(x * std::cos(theta))
                      * x_cutoff(x);
    }
};

asgard::pde_scheme<P> make(asgard::prog_opts options)
{
    options.title = "LHCD 2D quasilinear Fokker-Planck";
    P const Zi = options.file_required<P>("Zi");
    P const x_max = options.file_required<P>("x_max");

    LHCDDiffusion diffusion_coefficient{
        options.file_required<P>("cut_x_parallel_min"),
        options.file_required<P>("cut_x_parallel_max"),
        options.file_required<P>("cut_height"),
        options.file_value<P>("smoothing_width").value_or(P{0.02}),
        options.file_value<P>("cut_center").value_or(P{1.0}),
        x_max
    };

    if (!(diffusion_coefficient.x_parallel_min <
          diffusion_coefficient.x_parallel_max))
        throw std::runtime_error(
            "cut_x_parallel_min must be smaller than cut_x_parallel_max");
    if (diffusion_coefficient.smoothing_width < P{0})
        throw std::runtime_error("smoothing_width must be nonnegative");
    if (!(diffusion_coefficient.cut_center > P{0}))
        throw std::runtime_error("cut_center must be positive");
    if (!(x_max > P{0}))
        throw std::runtime_error("x_max must be positive");
    
    asgard::pde_domain<P> domain({{0.0, x_max}, {0.0, PI}});
    domain.set_names({"x", "theta"});
    
    // setting some default options
    // defaults are used only the corresponding values are missing from the command line
    options.default_degree = 2;
    options.default_start_levels = {5, 5};
    
    options.default_step_method = asgard::time_method::cn;
        
    options.default_solver = asgard::solver_method::gmres;
    options.default_precon = asgard::precon_method::none;
    
    options.default_isolver_tolerance  = 1.E-8;
    options.default_isolver_iterations = 1000;
    options.default_isolver_inner_iterations = 50;
    
    // create a pde from the given options and domain
    // we can read the variables using pde.options() and pde.domain() (both return const-refs)
    // the option entries may have been populated or updated with default values
    asgard::pde_scheme<P> pde(options, std::move(domain));
    // Spherical-coordinate mass M(x,theta)=x^2 sin(theta).
    auto mass_x = [](std::vector<P> const &x, std::vector<P> &value) {
        for (std::size_t i = 0; i < x.size(); ++i)
            value[i] = x[i] * x[i];
    };

    auto x_linear = [](std::vector<P> const &x, std::vector<P> &value) {
        for (std::size_t i = 0; i < x.size(); ++i)
            value[i] = x[i];
    };

    auto mass_th = [](std::vector<P> const &theta, std::vector<P> &value) {
        for (std::size_t i = 0; i < theta.size(); ++i)
            value[i] = std::sin(theta[i]);
    };

    auto neg_mass_th = [](std::vector<P> const &theta,
                          std::vector<P> &value) {
        for (std::size_t i = 0; i < theta.size(); ++i)
            value[i] = -std::sin(theta[i]);
    };
    
    pde.set_mass({term_volume{mass_x}, term_volume{mass_th}});
    
    // Collision terms: x components.
    {
        // Drift term: -1/x^2 d[x^2*(-1/2x^2)f]/dx.  bothsides: the drag
        // flux is fixed to zero at both radial walls (its boundary bracket
        // is omitted from the operator), which together with the closed
        // diffusive and QL fluxes below makes the wall exactly leakproof.
        pde += term_md({term_div(-0.5, asgard::flux_type::upwind,
                                 asgard::boundary_type::bothsides),
            term_identity{}});
        
        // Diffusion: s=1/x^2 d/dx(x^2 zeta q), D_c = 1/(4x^3) in the
        // standard sqrt(2T/m) units (see the normalization header).
        // (Formerly this chain carried set_*_robin(0.5), the counter-bracket
        // to the drag term's free boundary flux x^2 A f = f/2.  With the
        // drag term itself now bothsides, its bracket is deleted rather
        // than cancelled -- the assembled matrix is identical, verified
        // bit-for-bit on the 1-D solvers and to solver tolerance in 2-D.)
        term_1d div_grad_x({
            term_div(-0.25, asgard::flux_type::upwind,
                     asgard::boundary_type::bothsides),
            term_grad(x_linear),
        });

        P const inv_dx = P{1} / pde.cell_size(asgard::dimension_id{0});
        pde += term_md({div_grad_x, term_identity{}});
        pde += term_md({
            asgard::term_penalty<P>{inv_dx, asgard::boundary_type::none},
            term_identity{},
        });
    }
    
    // Collision terms: pitch-angle component.
    {
        term_1d div_grad_th(
                {term_div(neg_mass_th, asgard::flux_type::upwind,
                          asgard::boundary_type::bothsides),
                 term_grad(mass_th)}
        );

        // -(Zi+1)/(4x^3 sin(theta)) d/dtheta(sin(theta) df/dtheta).
        P const collision_factor = (Zi + P{1}) / P{4};
        term_1d collision_x({term_volume{x_linear},
                             term_volume{collision_factor}});
        
        pde += term_md({collision_x, div_grad_th});

        P const inv_dtheta = P{1} / pde.cell_size(asgard::dimension_id{1});
        term_1d penalty_th =
            asgard::term_penalty<P>{inv_dtheta, asgard::boundary_type::none};
        pde += term_md{{term_identity{}, penalty_th}};
    }
    
    // =====================================================================
    // LOWER-HYBRID QUASILINEAR DIFFUSION
    //
    // x_parallel=x cos(theta), so in the orthonormal spherical basis
    //
    //   e_parallel = cos(theta) e_x - sin(theta) e_theta,
    //
    // and D_orth = D_ql e_parallel e_parallel^T.  After multiplying by the
    // spherical mass M=x^2 sin(theta), the conservative coefficient matrix is
    //
    //   K = D_ql [ x^2 sin(theta) cos^2(theta),
    //             -x sin^2(theta) cos(theta);
    //             -x sin^2(theta) cos(theta),
    //              sin^3(theta) ].
    //
    // Every K_ij is separable apart from the common scalar D_ql(v cos theta).
    // Put the separable K_ij directly inside Grad_j and interpolate only the
    // common D_ql scalar.  This is the multidimensional Grad[q] construction:
    // the outer
    // divergence receives the global M^{-1}, while term_interp introduces no
    // additional mass operation.
    // =====================================================================
    // D_ql(v cos(theta)) is the only non-separable interpolated coefficient.
    // Its evaluation is just a few arithmetic operations, so evaluate it
    // directly instead of maintaining a node hash and value cache.
    auto D_adapt = [diffusion_coefficient](
                    P, asgard::vector2d<P> const &nodes,
                    std::vector<P> const &f, std::vector<P> &value) {
        for (std::int64_t i = 0; i < nodes.num_strips(); ++i) {
            P const x = nodes[i][0];
            P const theta = nodes[i][1];
            value[i] = std::abs(diffusion_coefficient.val(x, theta))
                     * std::abs(f[i]);
        }
    };
    pde.set_adapt_weight(D_adapt);

    auto D_ql = [diffusion_coefficient](
                    P, asgard::vector2d<P> const &nodes,
                    std::vector<P> const &input, std::vector<P> &output) {
        for (std::int64_t i = 0; i < nodes.num_strips(); ++i) {
            P const x = nodes[i][0];
            P const theta = nodes[i][1];
            output[i] = diffusion_coefficient.val(x, theta) * input[i];
        }
    };

    // Plain conservative divergences. The PDE mass applies
    // 1/(x^2 sin(theta)) to the completed flux divergence.
    // bc::bothsides closes the QL flux at the domain edges.  Required
    // together with the drag and diffusion brackets above being closed
    // (bothsides on both collisional terms).  The RF window is a
    // band in x_parallel, so it is still LIVE on part of the outer wall --
    // at x = x_max the resonance sits at cos(theta) = x_par/x, e.g. D_ql =
    // 0.5 at theta ~ 68 deg for x_max = 8.5 -- and cut_center >= 1 leaves it
    // unattenuated.  Left as bc::none (outflow) that flux simply escapes.
    //
    // Zeroing BOTH components is what makes the edge condition tractable:
    // D_ql = D_w e_par e_par^T is rank-1, so vanishing in both directions
    // forces e_par.grad(f) = 0 and the QL flux drops out entirely, leaving
    // the purely collisional B df/dx + A f = 0 that the Robin implements.
    // Otherwise Gamma_x couples df/dtheta and no scalar Robin exists.
    // (ICRF_2D carries the same fix; see its longer note there for the
    // measured failure modes -- Neumann manufactures particles, and the
    // Robin without this bleeds them.)
    term_md div_x({
        term_div{P{-1.0}, asgard::flux_type::upwind,
                 asgard::boundary_type::bothsides},
        term_volume{P{1.0}}
    });
    term_md div_theta({
        term_volume{P{1.0}},
        term_div{P{-1.0}, asgard::flux_type::upwind,
                 asgard::boundary_type::bothsides}
    });

    auto K_xx_theta = [](std::vector<P> const &theta,
                         std::vector<P> &value) {
        for (std::size_t i = 0; i < theta.size(); ++i) {
            P const sin_theta = std::sin(theta[i]);
            P const cos_theta = std::cos(theta[i]);
            value[i] = sin_theta * cos_theta * cos_theta;
        }
    };
    auto K_cross_theta = [](std::vector<P> const &theta,
                            std::vector<P> &value) {
        for (std::size_t i = 0; i < theta.size(); ++i) {
            P const sin_theta = std::sin(theta[i]);
            P const cos_theta = std::cos(theta[i]);
            value[i] = -sin_theta * sin_theta * cos_theta;
        }
    };
    auto K_thetatheta_theta = [](std::vector<P> const &theta,
                                 std::vector<P> &value) {
        for (std::size_t i = 0; i < theta.size(); ++i) {
            P const sin_theta = std::sin(theta[i]);
            value[i] = sin_theta * sin_theta * sin_theta;
        }
    };

    // K_xx f_x, K_xtheta f_theta, K_thetax f_x, K_thetatheta f_theta.
    // The two cross entries use exactly the same separable factors.
    term_md grad_K_xx({
        term_grad{mass_x},
        term_volume{K_xx_theta}
    });
    term_md grad_K_xtheta({
        term_volume{x_linear},
        term_grad{K_cross_theta}
    });
    term_md grad_K_thetax({
        term_grad{x_linear},
        term_volume{K_cross_theta}
    });
    term_md grad_K_thetatheta({
        term_volume{P{1.0}},
        term_grad{K_thetatheta_theta}
    });

    term_md diffusion = term_interp{D_ql};

    pde += term_md({div_x,     diffusion, grad_K_xx});
    pde += term_md({div_x,     diffusion, grad_K_xtheta});
    pde += term_md({div_theta, diffusion, grad_K_thetax});
    pde += term_md({div_theta, diffusion, grad_K_thetatheta});

    // initial condition f(0,x,theta) = f_0(x,theta), user-supplied as the
    // (fully 2-D, non-separable allowed) lambda in
    // input_data/initial_condition.hpp -- a compile input: editing it
    // triggers a rebuild and a re-solve through the normal CMake staleness
    // rules.
    {
        // Normalize numerically as a 3-D velocity distribution:
        // 2 pi * integral f0 x^2 sin(theta) dx dtheta = 1, so the default
        // Maxwellian shape reproduces the old hard-coded (2 pi)^{-3/2}
        // prefactor exactly.  Composite 2-D Simpson: milliseconds, and both
        // integrands here are smooth.
        P norm = 0;
        {
            int const nx = 2048, nth = 1024;  // intervals; even, per Simpson
            P const hx = x_max / nx;
            P const hth = PI / nth;
            for (int i = 0; i <= nx; ++i) {
                P const x = i * hx;
                P const wx = (i == 0 || i == nx) ? 1 : (i % 2 ? 4 : 2);
                P row = 0;
                for (int j = 0; j <= nth; ++j) {
                    P const th = j * hth;
                    P const wth = (j == 0 || j == nth) ? 1 : (j % 2 ? 4 : 2);
                    row += wth * initial_f0(x, th) * std::sin(th);
                }
                norm += wx * x * x * row * (hth / 3);
            }
            norm *= P{2} * PI * hx / 3;
            if (!(norm > 0) || !std::isfinite(norm))
                throw std::runtime_error(
                    "initial_condition.hpp: 2 pi * integral of "
                    "f0 x^2 sin(theta) must be positive and finite");
        }
        // Unlike the old separable path (which projected mass-weighted
        // values against the local mass matrices), the non-separable initial
        // condition goes through asgard's interpolation machinery, which
        // collocates the function *itself* -- so supply plain f0, no
        // Jacobian.
        pde.set_initial([norm](P, asgard::vector2d<P> const &nodes,
                               std::vector<P> &fx) {
            for (std::int64_t i = 0; i < nodes.num_strips(); ++i)
                fx[i] = initial_f0(nodes[i][0], nodes[i][1]) / norm;
        });
    }
    
    return pde;
}

int main(int argc, char** argv){
    // if MPI is enabled, call MPI_Init(), otherwise do nothing
    asgard::libasgard_runtime running_(argc, argv);
    
    // if double precision is available the P is double
    // otherwise P is float
    using P = asgard::default_precision;
    
    // parse the command-line inputs
    asgard::prog_opts options(argc, argv);

    if (options.show_help) {
        std::cout <<
          "\n LHCD 2D quasilinear Fokker-Planck solver\n"
          "   dim0 = x (=v/v_e), dim1 = theta (pitch angle, rad)\n"
          "   RF diffusion is localized in x_parallel=x cos(theta).\n\n"
          " Run with:\n"
          "     ./LHCD_2D -if input_solver.txt [ASGarD options]\n\n"
          "    -- standard ASGarD options --";
        options.print_help(std::cout);
        return 0;
    }

    // the discretization_manager takes in a pde and handles sparse-grid construction
    // separable and non-separable operators, holds the current state, etc.
    asgard::discretization_manager<P> disc(make(options),
                                           asgard::verbosity_level::high);

    int movie_stride = options.file_value<int>("movie_stride").value_or(0);
    int const movie_frames = options.file_value<int>("movie_frames").value_or(0);
    std::string const movie_dir =
        options.file_value<std::string>("movie_dir").value_or("movie");

    if (movie_stride <= 0 && movie_frames > 1) {
        int64_t const total_steps = disc.remaining_steps();
        movie_stride = static_cast<int>(
            std::max<int64_t>(1, (total_steps + movie_frames - 2) /
                                      (movie_frames - 1)));
    }

    bool success = true;
    if (movie_stride > 0) {
        std::filesystem::path const movie_path(movie_dir);
        std::filesystem::create_directories(movie_path);

        // Remove only snapshots created by an earlier LHCD run.
        for (auto const &entry : std::filesystem::directory_iterator(movie_path)) {
            if (!entry.is_regular_file()) continue;
            std::string const filename = entry.path().filename().string();
            std::string const ext = entry.path().extension().string();
            if (filename.rfind("snapshot_", 0) == 0 &&
                (ext == ".h5" || ext == ".hdf5"))
                std::filesystem::remove(entry.path());
        }

        int frame_index = 0;
        auto save_movie_snapshot = [&]() {
            std::ostringstream name;
            name << "snapshot_"
                 << std::setw(6) << std::setfill('0') << frame_index++
                 << "_step_"
                 << std::setw(8) << std::setfill('0') << disc.current_step()
                 << ".h5";
            disc.save_snapshot(movie_path / name.str());
        };

        save_movie_snapshot();
        while (disc.remaining_steps() > 0) {
            int64_t const chunk =
                std::min<int64_t>(movie_stride, disc.remaining_steps());
            success = disc.advance_time(chunk);
            save_movie_snapshot();
            if (!success) break;
        }
    } else {
        success = disc.advance_time();
    }

    disc.final_output();

    if (!success || disc.remaining_steps() > 0) {
        std::cerr << "LHCD_2D: time advancement stopped before the requested "
                     "final time.\n";
        return 2;
    }
    return 0;
};
