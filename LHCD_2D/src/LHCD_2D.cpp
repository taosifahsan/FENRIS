// LHCD 2-D quasilinear Fokker-Planck solver on (x, theta).
//
// Units: x = v/v_th, v_th = sqrt(2 T_e/m_e); time in collision times of that
//   v_th.  Equilibrium exp(-x^2).
// Coefficients: diffusion 1/(4x^3), drag 1/(2x^2), pitch (Zi+1)/(4x^3).
// Boundaries: bothsides on every outermost div -> zero total flux at both
//   walls; grads and penalties stay free.
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

// User-editable f0(x, theta); lives in input_data/ because it is an input.
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

// RF diffusion strength: a window in x_parallel = x cos(theta), times an
// optional high-speed cutoff in x/x_max.  Both edges share smoothing_width.
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
    
    // Defaults; used only when the deck / command line omits the value.
    options.default_degree = 2;
    options.default_start_levels = {5, 5};
    
    options.default_step_method = asgard::time_method::cn;
        
    options.default_solver = asgard::solver_method::gmres;
    options.default_precon = asgard::precon_method::none;
    
    options.default_isolver_tolerance  = 1.E-8;
    options.default_isolver_iterations = 1000;
    options.default_isolver_inner_iterations = 50;
    
    asgard::pde_scheme<P> pde(options, std::move(domain));

    // Spherical mass M(x,theta) = x^2 sin(theta), and its separable factors.
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
        // Drag: -1/x^2 d[x^2 (-1/2x^2) f]/dx.
        pde += term_md({term_div(-0.5, asgard::flux_type::upwind,
                                 asgard::boundary_type::bothsides),
            term_identity{}});

        // Speed diffusion: 1/x^2 d/dx(x^2 D_c df/dx), D_c = 1/(4x^3).
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
    
    // ---- Lower-hybrid quasilinear diffusion -----------------------------
    // D = D_ql e_par e_par^T with e_par = cos(theta) e_x - sin(theta) e_th.
    // Times the mass M, the coefficient matrix is
    //   K = D_ql [ x^2 s c^2, -x s^2 c ; -x s^2 c, s^3 ]   (s=sin, c=cos),
    // whose entries are all separable apart from the scalar D_ql.  So each
    // K_ij goes inside its Grad_j and only D_ql is interpolated.
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

    // Conservative divergences; the PDE mass supplies 1/(x^2 sin theta).
    // bothsides on BOTH components: the RF window is still live on part of
    // the outer wall, and zeroing both kills the rank-1 QL flux entirely,
    // leaving the collisional B df/dx + A f = 0 that the sealed drag and
    // diffusion impose between them.  Left open, that flux escapes.
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

    // Initial condition from input_data/initial_condition.hpp, normalized so
    // 2 pi * integral f0 x^2 sin(theta) dx dtheta = 1 (a 3-D distribution).
    // 2-D composite Simpson; both integrands are smooth.
    {
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
        // Interpolation collocates the function itself: pass plain f0, no
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
    asgard::libasgard_runtime running_(argc, argv);
    using P = asgard::default_precision;
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
